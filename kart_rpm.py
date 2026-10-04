#!/usr/bin/env python3
"""
kart_rpm.py - estimate engine RPM over time from the audio of a kart video.

Engine model
------------
A single-cylinder four-stroke fires once every two crank revolutions, so the
exhaust note is a harmonic series whose fundamental is

        f0 = RPM / 120        (1500 RPM -> 12.5 Hz,  6500 RPM -> 54.2 Hz)

with overtones at 2*f0, 3*f0, ... up to several hundred Hz.

Pipeline
--------
1. Decode the audio to mono 8 kHz (ffmpeg, or librosa as a fallback).
2. Short-time spectrum (0.4 s window, 0.05 s hop), log-compressed and
   "whitened" (local median subtracted) so harmonic peaks stand out from
   wind / tyre / broadband noise.
3. Engine on/off: the camera hears your own engine much louder than anything
   else, so a 3-segment step (off -> on -> off) is fitted to the 25-1000 Hz
   loudness curve. Other karts heard before start / after shutdown are ignored.
4. Harmonic salience: for every candidate RPM (10 RPM grid), average the
   whitened spectrum at the harmonics k*RPM/120.
5. Tracking: Viterbi (dynamic programming) finds the smoothest high-salience
   path, with a cap on how fast RPM can change. This bridges frames where the
   engine note is masked (braking, wind, other karts).
6. Plot RPM vs. time and save a CSV.

Usage
-----
    python kart_rpm.py onboard.mp4              # writes onboard_rpm.png + onboard_rpm.csv
    python kart_rpm.py onboard.mp4 --show       # also opens the plot window
    python kart_rpm.py onboard.mp4 --debug      # extra spectrogram + overlay image

Requires: numpy, scipy, matplotlib, and either ffmpeg on PATH or librosa.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import stft

SR = 8000            # analysis sample rate (Hz) - engine harmonics live < 1 kHz
WIN_S = 0.40         # STFT window (s): resolves harmonics 12.5 Hz apart at idle
HOP_S = 0.05         # STFT hop (s) -> 20 RPM readings per second
NFFT = 8192          # zero-padded FFT (~1 Hz bins)


# --------------------------------------------------------------------------- #
# 1. Audio loading
# --------------------------------------------------------------------------- #
def load_audio(path, sr=SR):
    """Return mono float32 samples at `sr`."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
               "-f", "f32le", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.float32).copy()
    try:
        import librosa
    except ImportError as e:
        raise SystemExit("Need either ffmpeg on PATH or `pip install librosa`.") from e
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Spectrogram
# --------------------------------------------------------------------------- #
def spectrogram(y, sr=SR, fmax=1000.0):
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    f, t, Z = stft(y, sr, window="hann", nperseg=win, noverlap=win - hop,
                   nfft=NFFT, boundary=None, padded=False)
    keep = f <= fmax
    f, mag = f[keep], np.abs(Z[keep])
    t = t  # centre time of each frame (s)

    # loudness in the band where the engine dominates (for on/off detection)
    band = (f >= 25) & (f <= 1000)
    loud_db = 10 * np.log10((mag[band] ** 2).mean(axis=0) + 1e-12)

    # whitening: log magnitude minus its running median over ~80 Hz
    S = np.log1p(mag * 1000.0)
    df = f[1] - f[0]
    W = np.clip(S - median_filter(S, size=(int(80 / df), 1)), 0, None)
    return f, t, W, loud_db, mag


# --------------------------------------------------------------------------- #
# 3. Engine on/off detection
# --------------------------------------------------------------------------- #
def _seg_cost(c1, c2, a, b):
    """Sum of squared errors of a constant fit to x[a:b] (vectorised)."""
    n = np.maximum(b - a, 1)
    s = c1[b] - c1[a]
    return (c2[b] - c2[a]) - s * s / n


def detect_engine_on(loud_db, hop_s=HOP_S):
    """Fit off -> on -> off to the loudness curve; return (i_on, i_off) frames."""
    x = uniform_filter1d(loud_db, max(1, int(1.0 / hop_s)))      # 1 s smoothing
    n = len(x)
    c1 = np.concatenate([[0], np.cumsum(x)])
    c2 = np.concatenate([[0], np.cumsum(x * x)])
    step = 2
    best = (np.inf, 0, n)
    for i in range(step, n - step, step):
        j = np.arange(i + step, n, step)
        cost = (_seg_cost(c1, c2, 0, i) + _seg_cost(c1, c2, i, j)
                + _seg_cost(c1, c2, j, n))
        k = int(np.argmin(cost))
        if cost[k] < best[0]:
            best = (cost[k], i, j[k])
    _, i_on, i_off = best

    # refine each edge with light smoothing (0.25 s) inside a +-6 s window
    x2 = uniform_filter1d(loud_db, max(1, int(0.25 / hop_s)))
    def refine(i):
        r = int(6.0 / hop_s)
        a, b = max(1, i - r), min(n - 1, i + r)
        seg = x2[max(0, a - r):min(n, b + r)]
        off = max(0, a - r)
        cs1 = np.concatenate([[0], np.cumsum(seg)])
        cs2 = np.concatenate([[0], np.cumsum(seg * seg)])
        cand = np.arange(a, b) - off
        cost = _seg_cost(cs1, cs2, 0, cand) + _seg_cost(cs1, cs2, cand, len(seg))
        return int(cand[np.argmin(cost)] + off)
    return refine(i_on), refine(i_off)


# --------------------------------------------------------------------------- #
# 4. Harmonic salience
# --------------------------------------------------------------------------- #
def harmonic_salience(W, f, rpm_grid, hmin=28.0, hmax=700.0):
    """Mean whitened magnitude at the harmonics of f0 = rpm/120.

    hmin skips frequencies the camera microphone filters out (~<30 Hz)."""
    df = f[1] - f[0]
    sal = np.empty((len(rpm_grid), W.shape[1]), np.float32)
    for i, rpm in enumerate(rpm_grid):
        h = np.arange(1, 100) * rpm / 120.0
        h = h[(h >= hmin) & (h <= hmax)]
        idx = h / df
        lo = np.floor(idx).astype(int)
        fr = (idx - lo)[:, None]
        sal[i] = (W[lo] * (1 - fr) + W[lo + 1] * fr).mean(axis=0)
    return sal


def octave_guard(rpm_grid, low_bias=1.3, ramp=(2800, 3400)):
    """Weight that slightly favours low RPM states.

    At idle a single-cylinder four-stroke is quiet and its loudest line is
    often 2*f0 (once per crank rev), which a plain harmonic average reads as
    double the true RPM. Boosting states below ~3000 RPM resolves this; when
    racing (RPM > ~3500) the octave-down alternative is out of range anyway."""
    lo, hi = ramp
    w = np.interp(rpm_grid, [lo, hi], [low_bias, 1.0])
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# 5. Viterbi tracking
# --------------------------------------------------------------------------- #
def viterbi_track(E, rpm_step, max_rate=5000.0, smooth=0.002, hop_s=HOP_S):
    """Best path through emission matrix E (states x frames).

    max_rate : largest allowed RPM change per second
    smooth   : penalty per (100 RPM)^2 of change between frames"""
    S, T = E.shape
    max_states = max(1, int(round(max_rate * hop_s / rpm_step)))
    steps = np.arange(-max_states, max_states + 1)
    pen = smooth * (steps * rpm_step / 100.0) ** 2 * 100

    D = E[:, 0].astype(np.float64).copy()
    back = np.zeros((S, T), np.int16)
    for k in range(1, T):
        best = np.full(S, -np.inf)
        arg = np.zeros(S, np.int16)
        for s, p in zip(steps, pen):
            cand = np.full(S, -np.inf)
            if s >= 0:
                cand[s:] = D[:S - s] - p
            else:
                cand[:S + s] = D[-s:] - p
            m = cand > best
            best[m] = cand[m]
            arg[m] = s
        D = best + E[:, k]
        back[:, k] = arg
    path = np.empty(T, int)
    path[-1] = int(np.argmax(D))
    for k in range(T - 1, 0, -1):
        path[k - 1] = path[k] - back[path[k], k]
    return path


# --------------------------------------------------------------------------- #
# 6. Plotting
# --------------------------------------------------------------------------- #
def _mmss(x, _pos=None):
    x = max(0, x)
    return f"{int(x // 60)}:{int(x % 60):02d}"


def plot_rpm(t, rpm, t_on, t_off, duration, out_png, title, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ink, ink2, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for a, b in [(0, t_on), (t_off, duration)]:
        ax.axvspan(a, b, color="#ecebe7", lw=0)
        ax.text((a + b) / 2, 0.97, "engine off", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=ink2, fontsize=9)
    ax.plot(t, rpm, color=blue, lw=1.2)

    ax.set_xlim(0, duration)
    ax.set_ylim(1000, 7000)
    ax.xaxis.set_major_locator(MultipleLocator(60 if duration > 240 else 15))
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_xlabel("Video time (m:ss)", color=ink2)
    ax.set_ylabel("Engine RPM", color=ink2)
    ax.set_title(title, loc="left", color=ink, fontsize=13)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b5b4ae")
    ax.tick_params(colors=ink2)

    on = ~np.isnan(rpm)
    if on.any():
        i = np.nanargmax(rpm)
        ax.annotate(f"max {rpm[i]:.0f}", (t[i], rpm[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", color=ink, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def plot_debug(f, t, W, rpm, out_png):
    """Whitened spectrogram with the tracked harmonics drawn on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    keep = f <= 400
    fig, ax = plt.subplots(figsize=(max(14, t[-1] / 15), 6))
    ax.pcolormesh(t, f[keep], W[keep], shading="auto", cmap="magma", vmax=np.percentile(W, 99.5))
    for k in (1, 2, 3, 4, 6, 8):
        ax.plot(t, rpm / 120 * k, color="#3bd1ff", lw=0.6, alpha=0.8)
    ax.set_ylim(0, 400)
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_xlabel("time (m:ss)")
    ax.set_ylabel("Hz")
    ax.set_title("Whitened spectrogram with tracked harmonics (k = 1,2,3,4,6,8 x RPM/120)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def analyze(path, rpm_min=1500, rpm_max=6500, rpm_step=10, max_rate=5000.0,
            whole=False):
    """whole=True tracks RPM over the entire clip instead of only the detected
    engine-on span (used for matching, where the telemetry itself says when
    the engine was running)."""
    y = load_audio(path)
    duration = len(y) / SR
    f, t, W, loud_db, _ = spectrogram(y)

    if whole:
        i_on, i_off = 0, len(t)
    else:
        i_on, i_off = detect_engine_on(loud_db)

    rpm_grid = np.arange(rpm_min, rpm_max + rpm_step, rpm_step, dtype=float)
    sal = harmonic_salience(W[:, i_on:i_off], f, rpm_grid)
    E = sal * octave_guard(rpm_grid)[:, None]
    path = viterbi_track(E, rpm_step, max_rate=max_rate)

    rpm = np.full(len(t), np.nan)
    rpm[i_on:i_off] = median_filter(rpm_grid[path], size=5, mode="nearest")
    return dict(t=t, rpm=rpm, t_on=float(t[i_on]), t_off=float(t[i_off - 1]) if whole else float(t[i_off]),
                duration=duration, f=f, W=W)


def main():
    ap = argparse.ArgumentParser(description="Engine RPM from kart video audio.")
    ap.add_argument("audio", help="audio file (mp3, m4a, wav, or even the video itself)")
    ap.add_argument("-o", "--out", help="output PNG (default: <audio>_rpm.png)")
    ap.add_argument("--csv", help="output CSV (default: <audio>_rpm.csv)")
    ap.add_argument("--rpm-min", type=float, default=1500)
    ap.add_argument("--rpm-max", type=float, default=6500)
    ap.add_argument("--max-rate", type=float, default=5000,
                    help="max RPM change per second allowed by the tracker")
    ap.add_argument("--show", action="store_true", help="open an interactive plot window")
    ap.add_argument("--debug", action="store_true",
                    help="also save a spectrogram with the tracked harmonics overlaid")
    a = ap.parse_args()

    base = os.path.splitext(a.audio)[0]
    out_png = a.out or base + "_rpm.png"
    out_csv = a.csv or base + "_rpm.csv"

    r = analyze(a.audio, a.rpm_min, a.rpm_max, max_rate=a.max_rate)
    print(f"Engine on  at {_mmss(r['t_on'])}  ({r['t_on']:.1f} s)")
    print(f"Engine off at {_mmss(r['t_off'])}  ({r['t_off']:.1f} s)")
    on = ~np.isnan(r["rpm"])
    print(f"RPM while running: min {np.nanmin(r['rpm']):.0f}, "
          f"median {np.nanmedian(r['rpm']):.0f}, max {np.nanmax(r['rpm']):.0f}")

    np.savetxt(out_csv, np.column_stack([r["t"][on], r["rpm"][on]]), delimiter=",",
               fmt=["%.2f", "%.0f"], header="time_s,rpm", comments="")
    plot_rpm(r["t"], r["rpm"], r["t_on"], r["t_off"], r["duration"], out_png,
             f"Engine RPM - {os.path.basename(a.audio)}", show=a.show)
    print(f"Saved {out_png} and {out_csv}")
    if a.debug:
        dbg = base + "_rpm_debug.png"
        plot_debug(r["f"], r["t"], r["W"], r["rpm"], dbg)
        print(f"Saved {dbg}")


if __name__ == "__main__":
    main()
