#!/usr/bin/env python3
"""Cache keypoints from the Firebird video using YOLOv8s-pose.

The previously cached `.keypoints_cache.npz` was built with yolov8n-pose
(nano, smallest/noisiest). This script re-runs detection with a larger,
more accurate pose model and saves a fresh cache.

Usage:
    python3 recache_keypoints.py --model yolov8s-pose.pt
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
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_MEDIA_DIR = os.environ.get("FIREBIRD_MEDIA_DIR")
VIDEO = os.environ.get("FIREBIRD_VIDEO")
if VIDEO is None and _MEDIA_DIR:
    VIDEO = str(Path(_MEDIA_DIR).expanduser() / "FIREBIRD CONDUCTOR BEELD_AUDIO.mov")
OUT = Path(__file__).parent / ".keypoints_cache_xl.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=VIDEO,
        required=VIDEO is None,
        help="Conductor video (or set FIREBIRD_VIDEO/FIREBIRD_MEDIA_DIR).",
    )
    parser.add_argument("--model", default="yolov8s-pose.pt",
                        help="YOLOv8 pose model (n/s/m/l/x)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference resolution (default 640)")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--device", default="mps",
                        help="Device: mps (Apple GPU), cpu, cuda")
    args = parser.parse_args()

    print(f"Loading model {args.model}... device={args.device}")
    model = YOLO(args.model)

    video = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total} frames @ {fps:.1f}fps {w}x{h}")

    keypoints = np.zeros((total, 17, 3), dtype=np.float32)
    n_tracked = 0

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=args.imgsz, verbose=False, conf=args.conf, device=args.device)
        if (results and results[0].keypoints is not None
                and len(results[0].keypoints.data) > 0
                and results[0].boxes is not None):
            all_kps = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(areas.argmax())  # biggest person = conductor
            keypoints[i] = all_kps[best]
            n_tracked += 1

        if i % 100 == 0:
            print(f"  {i}/{total} ({100*i/total:.0f}%) tracked={n_tracked}")

    cap.release()

    np.savez_compressed(OUT,
                        keypoints=keypoints,
                        fps=np.array(fps),
                        total_frames=np.array(total))
    print(f"\nSaved: {OUT}")
    print(f"  {n_tracked}/{total} frames tracked")


if __name__ == "__main__":
    main()
