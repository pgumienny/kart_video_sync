#!/usr/bin/env python3
"""
kart_sync.py - find which MyChron/AiM telemetry file matches a kart video,
and the time offset between them, by correlating engine RPM.

    python kart_sync.py onboard.mp4                 # searches ./telemetry/*.csv next to the video
    python kart_sync.py onboard.mp3 --dir logs      # extracted audio works too

Result:  telemetry_time = video_time - offset
         (i.e. telemetry t=0 happens `offset` seconds into the video)

How it works
------------
1. RPM is extracted from the video's audio with kart_rpm.analyze()
   (same folder), at 20 readings/s - the same rate the logger records.
2. Coarse search: for every CSV, a masked normalised cross-correlation of the
   two RPM traces is computed for every possible lag (via FFT). Only samples
   where both the engine is running (video) and RPM > 500 (logger) count, and
   at least --min-overlap seconds must overlap. Best file = highest peak.
3. Fine alignment for the winner: the video is split into 60 s windows and
   each window is aligned separately to 5 ms resolution. The median of the
   well-correlated windows is the final offset; a line through them gives
   clock drift between camera and logger.
4. Prints a ranking, writes <audio>_sync.png (overlay) and <audio>_sync.csv
   (video RPM + matching telemetry RPM per video timestamp).
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
from scipy.signal import fftconvolve

from kart_rpm import analyze, _mmss, HOP_S


# --------------------------------------------------------------------------- #
# Offset formatting
# --------------------------------------------------------------------------- #
def format_offset(seconds, style="clock"):
    """Format an offset as minutes / seconds / milliseconds.

    style="clock" -> "4:42.870"     (for display; negative: "-0:12.500")
    style="file"  -> "4m42s870ms"   (safe in file names on every OS)"""
    sign = "-" if seconds < 0 else ""
    ms_total = int(round(abs(seconds) * 1000))
    m, rem = divmod(ms_total, 60_000)
    s, ms = divmod(rem, 1000)
    if style == "file":
        return f"{sign}{m}m{s:02d}s{ms:03d}ms"
    return f"{sign}{m}:{s:02d}.{ms:03d}"


# --------------------------------------------------------------------------- #
# Telemetry loading (AiM / MyChron CSV export)
# --------------------------------------------------------------------------- #
def load_aim_csv(path, channel="RPM"):
    """Return (metadata dict, time array [s], channel array)."""
    meta, rows, header = {}, [], None
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.reader(fh)
        for r in rd:
            if not r:
                continue
            if header is None:
                if r[0] == "Time" and channel in r:
                    header = r
                    next(rd, None)            # units row
                else:
                    meta[r[0]] = r[1] if len(r) > 1 else ""
                continue
            rows.append((r[0], r[header.index(channel)]))
    if header is None:
        raise ValueError(f"{path}: no '{channel}' column")
    a = np.array(rows, float)
    return meta, a[:, 0], a[:, 1]


def resample(t, x, dt):
    """Put telemetry on a uniform grid with step dt (it's usually already 20 Hz)."""
    tg = np.arange(t[0], t[-1] + dt / 2, dt)
    return tg, np.interp(tg, t, x)


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def masked_xcorr(x, mx, y, my):
    """Pearson correlation of y slid along x, using only samples where both masks are 1.

    Returns lags (index of x aligned with y[0]), r, and overlap count N."""
    c = lambda a, b: fftconvolve(a, b[::-1], mode="full")
    N = c(mx, my)
    Sx, Sy = c(x * mx, my), c(mx, y * my)
    Sxx, Syy = c(x * x * mx, my), c(mx, y * y * my)
    Sxy = c(x * mx, y * my)
    with np.errstate(all="ignore"):
        cov = Sxy - Sx * Sy / N
        r = cov / np.sqrt((Sxx - Sx ** 2 / N) * (Syy - Sy ** 2 / N))
    lags = np.arange(-(len(y) - 1), len(x))
    return lags, np.nan_to_num(np.round(r, 12), nan=-1.0), np.round(N)


def coarse_match(tv, rv, tt, rt, dt, min_overlap_s):
    """Best offset (s) of telemetry t=0 within the video, its r and overlap (s)."""
    mv = ~np.isnan(rv)
    y, my = np.where(mv, rv, 0.0), mv.astype(float)
    mx = (rt > 500).astype(float)
    x = rt * mx
    lags, r, N = masked_xcorr(x, mx, y, my)
    r[N < min_overlap_s / dt] = -1
    k = int(np.argmax(r))
    # x[lag + i] ~ y[i]  ->  telemetry time = video time + lag*dt (+ t offsets)
    offset = tv[0] - (tt[0] + lags[k] * dt)
    return offset, float(r[k]), float(N[k] * dt)


def fine_align(tv, rv, tt, rt, offset0, win=60.0, search=1.5, step=0.005):
    """Align 60 s windows individually around offset0; return list of (t_mid, offset, r)."""
    ok = ~np.isnan(rv)
    rt = np.where(rt > 500, rt, np.nan)        # logger off / engine stopped
    cands = np.arange(offset0 - search, offset0 + search + step / 2, step)
    out = []
    t0, t1 = tv[ok][0], tv[ok][-1]
    for lo in np.arange(t0, t1 - win / 2, win / 2):
        s = ok & (tv >= lo) & (tv < lo + win)
        if s.sum() < win / HOP_S * 0.5:
            continue
        y = rv[s]
        if np.std(y) < 300:              # idle / flat sections carry no timing info
            continue
        best = (-1.0, offset0)
        for o in cands:
            x = np.interp(tv[s] - o, tt, rt, left=np.nan, right=np.nan)
            k = ~np.isnan(x)
            if k.sum() < 0.8 * len(x) or np.std(x[k]) == 0:
                continue
            c = np.corrcoef(x[k], y[k])[0, 1]
            if c > best[0]:
                best = (c, o)
        if best[0] > 0:
            out.append((lo + win / 2, best[1], best[0]))
    return out


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_sync(tv, rv, tt, rt, offset, name, out_png, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ink, ink2, grid, blue, orange = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#fcfcfb",
                             gridspec_kw=dict(height_ratios=[1, 1]))
    # zoom window: the 60 s with the most RPM variation where BOTH traces exist
    xt = np.interp(tv - offset, tt, np.where(rt > 500, rt, np.nan), left=np.nan, right=np.nan)
    ok = ~np.isnan(rv) & ~np.isnan(xt)
    w = min(int(60 / HOP_S), len(tv) - 1)
    starts = range(0, max(1, len(tv) - w), max(1, w // 4))
    stds = [np.std(xt[i:i + w]) if ok[i:i + w].all() else -1 for i in starts]
    i0 = list(starts)[int(np.argmax(stds))] if stds and max(stds) > 0 else 0
    zoom = (tv[i0], tv[min(i0 + w, len(tv) - 1)])

    for ax, xl, title in [(axes[0], (0, tv[-1]), f"Video RPM vs telemetry {name}  (offset {format_offset(offset)})"),
                          (axes[1], zoom, "Zoom")]:
        ax.set_facecolor("#fcfcfb")
        ax.plot(tv, rv, color=blue, lw=1.3, label="Video (from audio)")
        ax.plot(tt + offset, rt, color=orange, lw=1.0, label=f"Telemetry {name}")
        ax.set_xlim(*xl)
        ax.set_ylim(0, 7000)
        span = xl[1] - xl[0]
        ax.xaxis.set_major_locator(MultipleLocator(60 if span > 240 else 10))
        ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
        ax.set_ylabel("RPM", color=ink2)
        ax.set_title(title, loc="left", color=ink, fontsize=12)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b5b4ae")
        ax.tick_params(colors=ink2)
    axes[0].legend(loc="lower center", frameon=False, ncol=2, labelcolor=ink)
    axes[1].set_xlabel("Video time (m:ss)", color=ink2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def natural_key(p):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(p))]


def main():
    ap = argparse.ArgumentParser(description="Match kart video to telemetry by RPM.")
    ap.add_argument("video", help="video or extracted audio (mp4, mp3, m4a, wav...)")
    ap.add_argument("--dir", default=None, help="telemetry folder (default: telemetry/ next to the video)")
    ap.add_argument("--min-overlap", type=float, default=120, help="min overlap in seconds")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    tel_dir = a.dir or os.path.join(os.path.dirname(os.path.abspath(a.video)), "telemetry")
    files = sorted(glob.glob(os.path.join(tel_dir, "*.csv")), key=natural_key)
    if not files:
        raise SystemExit(f"No CSV files in {tel_dir}")

    print(f"Extracting RPM from {a.video} ...")
    res = analyze(a.video, whole=True)   # telemetry tells us when the engine ran
    tv, rv = res["t"], res["rpm"]
    dt = HOP_S

    print(f"Correlating against {len(files)} telemetry files ...\n")
    results = []
    for p in files:
        try:
            meta, t, r = load_aim_csv(p)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        tt, rt = resample(t, r, dt)
        off, rr, ov = coarse_match(tv, rv, tt, rt, dt, a.min_overlap)
        results.append(dict(path=p, name=os.path.basename(p), meta=meta, tt=tt, rt=rt,
                            offset=off, r=rr, overlap=ov))
    results.sort(key=lambda d: -d["r"])

    print(f"{'file':<10}{'date':<32}{'time':<10}{'corr':>6}{'offset':>11}{'overlap s':>11}")
    for d in results:
        m = d["meta"]
        print(f"{d['name']:<10}{m.get('Date', ''):<32}{m.get('Time', ''):<10}"
              f"{d['r']:>6.3f}{format_offset(d['offset']):>11}{d['overlap']:>11.0f}")

    best = results[0]
    margin = best["r"] - (results[1]["r"] if len(results) > 1 else 0)

    wins = fine_align(tv, rv, best["tt"], best["rt"], best["offset"])
    good = [w for w in wins if w[2] >= 0.9] or wins
    if good:
        offs = np.array([w[1] for w in good])
        offset = float(np.median(offs))
        spread = float(np.median(np.abs(offs - offset)))
        drift = (np.polyfit([w[0] for w in good], offs, 1)[0] * 1000) if len(good) >= 3 else float("nan")
    else:
        offset, spread, drift = best["offset"], float("nan"), float("nan")

    m = best["meta"]
    print("\n" + "=" * 64)
    print(f"Best match : {best['name']}  ({m.get('Session', '')}, {m.get('Date', '')} {m.get('Time', '')})")
    print(f"Correlation: {best['r']:.3f}   (next best file {best['r'] - margin:.3f})")
    if margin < 0.05:
        print("WARNING    : match is not clearly better than the runner-up - check the plot")
    print(f"Offset     : {format_offset(offset)}  (m:ss.mmm = {offset:+.3f} s, "
          f"+/- {spread * 1000:.0f} ms across {len(good)} windows)")
    print(f"             telemetry_time = video_time {'-' if offset >= 0 else '+'} {abs(offset):.3f}")
    if offset >= 0:
        print(f"             telemetry t=0 is {format_offset(offset)} into the video")
    else:
        print(f"             video t=0 is {format_offset(-offset)} into the telemetry")
    if not np.isnan(drift):
        print(f"Clock drift: {drift:+.1f} ms per 1000 s of video")
    print("=" * 64)

    base = os.path.splitext(a.video)[0]
    out_png, out_csv = base + "_sync.png", base + "_sync.csv"
    tel_on_video = np.interp(tv - offset, best["tt"], best["rt"], left=np.nan, right=np.nan)
    np.savetxt(out_csv, np.column_stack([tv, tv - offset, rv, tel_on_video]), delimiter=",",
               fmt="%.3f", header="video_time_s,telemetry_time_s,video_rpm,telemetry_rpm",
               comments="")
    plot_sync(tv, rv, best["tt"], best["rt"], offset, best["name"], out_png, show=a.show)
    print(f"Saved {out_png} and {out_csv}")


if __name__ == "__main__":
    main()
