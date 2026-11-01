#!/usr/bin/env python3
"""Compute the LDA-selected feature mask used by RF v2.0 production.

Builds the rich 792-feature training set, fits LDA, picks the top K
features by |coefficient|, and saves the mask + names to
analysis_results/lda_mask.npz so live_detector.py can load it.

This is a one-shot computation. Re-run after retraining if cues change.
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


import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from live_detector import ANNOT, CACHE, OUT, WARMUP_SEC
from pca_rich import (
    JOINT_STAT_NAMES, CROSS_NAMES, JOINTS, WINDOWS,
    cue_feature_vector,
)
from rf_v2_rich import build_rich_dataset

K_TOP = 120


def main():
    print("Loading data and building rich dataset…")
    data = np.load(CACHE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])
    cues = json.loads(Path(ANNOT).read_text())["cues"]
    cue_frames = np.array([int(c["time"] * fps) for c in cues])

    rng = np.random.default_rng(42)
    X, y, _, _, fnames = build_rich_dataset(kps, fps, total, cue_frames, rng)
    print(f"  X: {X.shape}, y pos={y.sum()} neg={(y==0).sum()}")

    print("Fitting LDA…")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(Xs, y)
    coef = lda.coef_[0]
    print(f"  LDA training accuracy: {lda.score(Xs, y):.3f}")

    rank = np.argsort(-np.abs(coef))
    top_indices = rank[:K_TOP].astype(np.int32)

    print(f"\nTop-15 selected features:")
    for r in range(15):
        idx = top_indices[r]
        print(f"  {r+1:>3}. {coef[idx]:+8.4f}   {fnames[idx]}")

    out_path = OUT / "lda_mask.npz"
    selected_names = np.array([fnames[i] for i in top_indices], dtype="U64")
    np.savez_compressed(
        out_path,
        indices=top_indices,
        coef=np.asarray([coef[i] for i in top_indices], dtype=np.float32),
        selected_names=selected_names,
        K=K_TOP,
    )
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
