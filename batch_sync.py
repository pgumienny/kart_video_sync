#!/usr/bin/env python3
"""
batch_sync.py - match every kart video in a folder to its telemetry file.

    python batch_sync.py "path/to/race folder"

For each video (*.MP4 / *.mov) in the folder:
  1. extract engine RPM from the audio            (kart_rpm.analyze)
  2. correlate it with every CSV in <folder>/telemetry   (kart_sync)
  3. verify the best candidates window-by-window
  4. if it's a confident match, copy the CSV to
        <folder>/<video name> - <offset>.csv      e.g. "GX010034 - 0m33s925ms.csv"
     where offset = time into the video at which telemetry t=0 occurs, as
     <minutes>m<seconds>s<milliseconds>ms (telemetry_time = video_time - offset).

Also writes <folder>/sync_check/<video> - sync.png (overlay plot) and
<folder>/sync_check/summary.csv. Needs kart_rpm.py and kart_sync.py next to it,
plus ffmpeg on PATH (or librosa).
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kart_rpm import analyze, HOP_S                                   # noqa: E402
from kart_sync import (load_aim_csv, resample, coarse_match,          # noqa: E402
                       fine_align, plot_sync, natural_key, format_offset)

VIDEO_EXT = (".mp4", ".mov", ".m4v")
MIN_OVERLAP_S = 120         # minimum video/telemetry overlap for a match
TOP_K = 4                   # coarse candidates that get the fine check
GOOD_WIN_R = 0.90           # a 60 s window "agrees" if it correlates above this
MIN_GOOD_FRAC = 0.40        # ...and at least this fraction of windows must agree
MAX_OFFSET_MAD = 0.25       # ...on an offset consistent to within this many seconds
AMBIGUOUS_GAP = 0.02        # flag if the runner-up correlates nearly as well


def verify(tv, rv, cand):
    """Window-by-window check of a coarse candidate. Returns a dict or None."""
    wins = fine_align(tv, rv, cand["tt"], cand["rt"], cand["offset"])
    if len(wins) < 2:
        return None
    offs = np.array([w[1] for w in wins])
    rs = np.array([w[2] for w in wins])
    good = rs >= GOOD_WIN_R
    use = offs[good] if good.sum() >= 2 else offs
    offset = float(np.median(use))
    mad = float(np.median(np.abs(use - offset)))
    # correlation over the whole overlap at the refined offset
    x = np.interp(tv - offset, cand["tt"], np.where(cand["rt"] > 500, cand["rt"], np.nan),
                  left=np.nan, right=np.nan)
    k = ~np.isnan(x) & ~np.isnan(rv)
    full_r = float(np.corrcoef(x[k], rv[k])[0, 1]) if k.sum() > 10 else 0.0
    return dict(full_r=full_r, overlap=float(k.sum() * HOP_S), offset=offset, mad=mad, good_frac=float(good.mean()),
                win_r=float(np.median(rs)), n_win=len(wins),
                ok=bool(good.mean() >= MIN_GOOD_FRAC and mad <= MAX_OFFSET_MAD))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--telemetry", default=None, help="default: <folder>/telemetry")
    a = ap.parse_args()

    folder = os.path.abspath(a.folder)
    tel_dir = a.telemetry or os.path.join(folder, "telemetry")
    check_dir = os.path.join(folder, "sync_check")
    os.makedirs(check_dir, exist_ok=True)

    videos = sorted([p for p in glob.glob(os.path.join(folder, "*"))
                     if p.lower().endswith(VIDEO_EXT)], key=natural_key)
    tel_files = sorted(glob.glob(os.path.join(tel_dir, "*.csv")), key=natural_key)
    print(f"{len(videos)} videos, {len(tel_files)} telemetry files\n")

    telemetry = []
    for p in tel_files:
        try:
            meta, t, r = load_aim_csv(p)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        tt, rt = resample(t, r, HOP_S)
        telemetry.append(dict(path=p, name=os.path.basename(p), meta=meta, tt=tt, rt=rt))

    summary = []
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0]
        t0 = time.time()
        print(f"=== {os.path.basename(vp)}  ({os.path.getsize(vp) / 1e9:.1f} GB)")
        row = dict(video=os.path.basename(vp), telemetry="", offset="", offset_s="", corr="",
                   offset_mad_s="", duration_s="", status="")
        try:
            res = analyze(vp, whole=True)
        except Exception as e:
            print(f"    audio extraction failed: {e}\n")
            row["status"] = f"audio error: {e}"
            summary.append(row)
            continue
        tv, rv = res["t"], res["rpm"]
        row["duration_s"] = f"{res['duration']:.0f}"
        print(f"    audio: {res['duration'] / 60:.1f} min ({time.time() - t0:.0f} s to analyse)")
        if res["duration"] < MIN_OVERLAP_S:
            print("    clip too short to match - skipped\n")
            row["status"] = "too short"
            summary.append(row)
            continue

        cands = []
        for tel in telemetry:
            off, r, ov = coarse_match(tv, rv, tel["tt"], tel["rt"], HOP_S, MIN_OVERLAP_S)
            cands.append(dict(tel, offset=off, r=r, overlap=ov))
        cands.sort(key=lambda c: -c["r"])

        verified = []
        for c in cands[:TOP_K]:
            v = verify(tv, rv, c)
            if v is None:
                continue
            c.update(v)
            print(f"    {c['name']:<8} coarse r={c['r']:.3f}  windows agree {v['good_frac']:>4.0%}  "
                  f"full r={v['full_r']:.3f}  offset {format_offset(v['offset']):>10} +/- {v['mad'] * 1000:.0f} ms"
                  f"{'  (candidate)' if v['ok'] else ''}")
            if v["ok"]:
                verified.append(c)

        if not verified:
            print("    no confident match\n")
            row["status"] = "no match"
            summary.append(row)
            continue

        verified.sort(key=lambda c: -c["full_r"])
        best = verified[0]
        status = "matched"
        if len(verified) > 1 and best["full_r"] - verified[1]["full_r"] < AMBIGUOUS_GAP:
            status = f"check: {verified[1]['name']} almost as good"

        off_txt = format_offset(best["offset"], style="file")
        out = os.path.join(folder, f"{stem} - {off_txt}.csv")
        shutil.copy2(best["path"], out)
        plot_sync(tv, rv, best["tt"], best["rt"], best["offset"], best["name"],
                  os.path.join(check_dir, f"{stem} - sync.png"))
        m = best["meta"]
        print(f"    -> {best['name']} ({m.get('Session', '')}, {m.get('Date', '')} "
              f"{m.get('Time', '')}), offset {format_offset(best['offset'])}"
              f"{'' if status == 'matched' else '   [' + status + ']'}")
        print(f"    wrote {os.path.basename(out)}\n")
        row.update(telemetry=best["name"], offset=format_offset(best["offset"]),
                   offset_s=f"{best['offset']:.3f}", corr=f"{best['full_r']:.3f}",
                   offset_mad_s=f"{best['mad']:.3f}", status=status)
        summary.append(row)

    with open(os.path.join(check_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()) if summary else ["video"])
        w.writeheader()
        w.writerows(summary)

    print("Summary")
    for r in summary:
        print(f"  {r['video']:<16} {r['status']:<10} {r['telemetry']:<8} {r['offset']}")
    used = [r["telemetry"] for r in summary if r["telemetry"]]
    dup = {u for u in used if used.count(u) > 1}
    if dup:
        print(f"  note: {', '.join(sorted(dup))} matched more than one video "
              f"(a session split across clips is normal)")
    print(f"\nCheck plots and summary.csv are in {check_dir}")


if __name__ == "__main__":
    main()
