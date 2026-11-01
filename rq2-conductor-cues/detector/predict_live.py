#!/usr/bin/env python3
"""Stream-style live inference using the trained cue detector.

Loads `analysis_results/live_detector_model.joblib` and walks through the
keypoint cache one frame at a time, computing the current causal feature
vector and emitting a "cue!" message whenever the smoothed score peaks above
threshold. This is the deployable form of the model: at each time step it
only looks at past frames.

Usage:
    python3 predict_live.py
    python3 predict_live.py --threshold 0.35
    python3 predict_live.py --start-sec 60 --end-sec 90
"""

from __future__ import annotations

# Resolve shared imports from this public repository layout.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
for _layout_root in _LayoutPath(__file__).resolve().parents:
    if (_layout_root / "shared" / "project_paths.py").is_file():
        _layout_sys.path.insert(0, str(_layout_root / "shared"))
        break
else:
    raise RuntimeError("Cannot locate shared/project_paths.py")
from project_paths import ROOT as _REPO_ROOT, AUDIO_ROOT as _AUDIO_ROOT, activate as _activate_layout, script_path as _script_path
_activate_layout()


import argparse
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from live_detector import (
    CACHE, ANNOT, OUT,
    JOINTS, WINDOWS, SMOOTH_K, REFRACTORY_SEC, WARMUP_SEC,
    extract_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="override threshold (default: model default)")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--model", type=str,
                    default=str(OUT / "live_detector_model.joblib"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    refractory_sec = bundle["refractory_sec"]

    data = np.load(CACHE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])

    cues = json.loads(Path(ANNOT).read_text())["cues"]
    cue_frames = np.array([int(c["time"] * fps) for c in cues])

    start_f = max(int(WARMUP_SEC * fps), int(args.start_sec * fps))
    end_f = total if args.end_sec is None else min(total, int(args.end_sec * fps))

    print(f"Streaming frames {start_f}..{end_f} "
          f"({(end_f-start_f)/fps:.1f}s) at {fps} fps, threshold={threshold}")

    refractory = int(refractory_sec * fps)
    smooth_buf: deque[float] = deque(maxlen=SMOOTH_K)
    prev_smooth = 0.0
    prev_prev_smooth = 0.0
    last_fire = -10**9
    fires: list[tuple[int, float]] = []

    for f in range(start_f, end_f):
        feat = extract_features(kps, f, fps, total)
        prob = float(model.predict_proba(feat[None, :])[0, 1])
        smooth_buf.append(prob)
        smooth = sum(smooth_buf) / len(smooth_buf)

        # Causal peak detection: rising→falling above threshold
        if (
            prev_smooth > threshold
            and prev_smooth >= prev_prev_smooth
            and prev_smooth >= smooth
            and (f - 1 - last_fire) >= refractory
        ):
            fires.append((f - 1, prev_smooth))
            last_fire = f - 1
            t = (f - 1) / fps
            mins = int(t // 60)
            secs = t - 60 * mins
            # Closest annotation
            diffs = np.abs(cue_frames - (f - 1))
            best = int(np.argmin(diffs))
            offset = (f - 1 - cue_frames[best]) / fps
            tag = ("CUE  " + cues[best]["label"][:24]
                   + f"  ({offset:+.2f}s)") if diffs[best] <= int(1.0 * fps) else "FALSE FIRE"
            print(f"  t={mins:02d}:{secs:05.2f}  p={prev_smooth:.3f}  →  {tag}")

        prev_prev_smooth = prev_smooth
        prev_smooth = smooth

    # Compare against ground truth in the streamed range
    gt_in_range = [int(cf) for cf in cue_frames if start_f <= cf < end_f]
    fire_frames = [f for f, _ in fires]
    tp = 0
    matched = set()
    for cf in gt_in_range:
        if fire_frames:
            diffs = np.abs(np.array(fire_frames) - cf)
            best = int(np.argmin(diffs))
            if diffs[best] <= int(1.0 * fps):
                tp += 1
                matched.add(best)
    fp = len(fire_frames) - len(matched)
    print()
    print(f"Streamed {(end_f-start_f)/fps:.1f}s, ground truth cues in range: {len(gt_in_range)}")
    print(f"  detected: {tp}/{len(gt_in_range)}, false fires: {fp}")


if __name__ == "__main__":
    main()
