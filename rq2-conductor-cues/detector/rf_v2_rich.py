#!/usr/bin/env python3
"""RF v2 on the 792-feature rich set, with full-timeline LOO and LDA check.

Compares against the 84-feature production model.

Pipeline:
  1. Extract 792 rich features for positives (±0.2s around 22 cues) and
     random negatives (≥1.5s away from any cue) — same sampling rule as
     the production dataset.
  2. Run LDA for an interpretable baseline on the rich set.
  3. Train a Random Forest with the same hyperparams as production.
  4. Full-timeline LOO: train without each cue, score the entire video,
     stitch frame-by-frame to the closest held-out cue's model.
  5. Report recall / fp/min / latency at multiple thresholds.

Output:
  analysis_results/rf_v2_rich_report.txt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from live_detector import (
    ANNOT, CACHE, OUT,
    DETECT_TOL_SEC, REFRACTORY_SEC, SMOOTH_K, WARMUP_SEC,
    causal_smooth, detect_peaks,
)

# Reuse rich feature extractor
from pca_rich import (
    JOINT_STAT_NAMES, CROSS_NAMES, JOINTS,
    cue_feature_vector,
)


def build_rich_dataset(kps, fps, total, cue_frames, rng,
                       pos_tol_sec=0.2, neg_margin_sec=1.5,
                       neg_per_sec=5, neg_ratio=6):
    pos_tol = int(pos_tol_sec * fps)
    neg_margin = int(neg_margin_sec * fps)
    warmup = int(WARMUP_SEC * fps)

    pos_samples = []
    for ci, cf in enumerate(cue_frames):
        for off in range(-pos_tol, pos_tol + 1):
            f = int(cf + off)
            if warmup <= f < total:
                pos_samples.append((f, 1, ci))

    skip = max(1, int(fps / neg_per_sec))
    candidates = np.arange(warmup, total, skip)
    far_mask = np.array([
        np.min(np.abs(cue_frames - f)) > neg_margin for f in candidates
    ])
    neg_pool = candidates[far_mask]
    n_neg_target = min(len(neg_pool), len(pos_samples) * neg_ratio)
    neg_idx = rng.choice(len(neg_pool), n_neg_target, replace=False)
    neg_samples = [(int(neg_pool[i]), 0, -1) for i in neg_idx]
    samples = pos_samples + neg_samples
    print(f"  positives: {len(pos_samples)}, negatives: {len(neg_samples)}")

    X = []
    feat_names = None
    for f, _, _ in samples:
        v, names = cue_feature_vector(kps, f / fps, fps, total)
        X.append(v)
        if feat_names is None:
            feat_names = names
    X = np.array(X)
    y = np.array([lbl for _, lbl, _ in samples], dtype=int)
    cue_ids = np.array([cid for _, _, cid in samples], dtype=int)
    frames = np.array([f for f, _, _ in samples], dtype=int)
    return X, y, cue_ids, frames, feat_names


def evaluate_loo(kps, fps, total, cue_frames, X, y, cue_ids, frames,
                 full_features, full_idx, threshold,
                 confirm_drop=0.05, max_watch_sec=0.6):
    n_cues = len(cue_frames)
    detect_tol = int(DETECT_TOL_SEC * fps)
    refractory = int(REFRACTORY_SEC * fps)
    all_scores = np.zeros((n_cues, len(full_idx)))
    detections = []
    for ho in range(n_cues):
        cf = cue_frames[ho]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask &= cue_ids != ho
        too_close = (cue_ids == -1) & (np.abs(frames - cf) <= detect_tol + refractory)
        train_mask &= ~too_close
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf.fit(X[train_mask], y[train_mask])
        all_scores[ho] = clf.predict_proba(full_features)[:, 1]

    closest = np.argmin(np.abs(full_idx[:, None] - cue_frames[None, :]),
                        axis=1)
    stitched = np.array([all_scores[ho, i] for i, ho in enumerate(closest)])
    smoothed = causal_smooth(stitched, SMOOTH_K)
    peaks = detect_peaks(smoothed, fps, threshold)
    peak_frames = [int(full_idx[p]) for p in peaks]

    tp = 0
    matched = set()
    latencies = []
    for cf in cue_frames:
        if peak_frames:
            d = np.array([abs(p - cf) for p in peak_frames])
            j = int(np.argmin(d))
            if d[j] <= detect_tol:
                tp += 1
                matched.add(j)
                latencies.append((peak_frames[j] - cf) / fps)
    fp = len(peak_frames) - len(matched)
    return {
        "recall": tp / n_cues,
        "tp": tp, "fp": fp,
        "fp_per_min": fp / (total / fps) * 60,
        "lat_mean": float(np.mean(latencies)) if latencies else 0.0,
        "lat_median": float(np.median(latencies)) if latencies else 0.0,
        "lat_abs": float(np.mean(np.abs(latencies))) if latencies else 0.0,
        "smoothed": smoothed,
    }


def main():
    print("Loading…")
    data = np.load(CACHE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])
    cues = json.loads(Path(ANNOT).read_text())["cues"]
    cue_frames = np.array([int(c["time"] * fps) for c in cues])

    print("Building rich dataset…")
    rng = np.random.default_rng(42)
    X, y, cue_ids, frames, fnames = build_rich_dataset(
        kps, fps, total, cue_frames, rng)
    print(f"  X: {X.shape}")

    # ── LDA on rich set ──────────────────────────────────────────────
    print("\nLDA on 792 features…")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(Xs, y)
    coef = lda.coef_[0]
    print(f"  LDA training accuracy: {lda.score(Xs, y):.3f}")

    order = np.argsort(-np.abs(coef))
    print(f"\n  Top-15 LDA features (signed):")
    for ix in order[:15]:
        print(f"    {coef[ix]:+8.4f}   {fnames[ix]}")

    # Category breakdown
    print(f"\n  LDA mass by stat keyword (Σ |coef|):")
    for kw in ["stillness", "speed_mean", "speed_max", "speed_std",
               "accel_mean", "accel_max", "jerk_mean", "jerk_max",
               "up_peak", "down_peak", "left_peak", "right_peak",
               "y_range", "x_range", "path_length", "straightness",
               "disp_y", "disp_x", "disp_mag",
               "end_speed", "spread"]:
        idxs = [i for i, n in enumerate(fnames)
                if kw in n.split("/")[-1] or (kw == "spread" and "spread" in n)]
        s = float(np.abs(coef[idxs]).sum()) if idxs else 0.0
        print(f"    {kw:<14} {s:>7.3f}  ({len(idxs)} feats)")

    print(f"\n  LDA mass by window:")
    for w in ["deep_pre", "prep", "now"]:
        idxs = [i for i, n in enumerate(fnames) if n.startswith(w + "/")]
        s = float(np.abs(coef[idxs]).sum())
        print(f"    {w:<10} {s:>7.3f}")

    # ── Pre-extract full-timeline rich features ──────────────────────
    print("\nExtracting full-timeline rich features (this is the slow part)…")
    full_idx = np.arange(int(WARMUP_SEC * fps), total)
    full_features = []
    for f in full_idx:
        v, _ = cue_feature_vector(kps, int(f) / fps, fps, total)
        full_features.append(v)
    full_features = np.array(full_features)
    print(f"  full_features: {full_features.shape}")

    # ── Full-timeline LOO ────────────────────────────────────────────
    print("\nFull-timeline LOO with RF (792 features)…")
    sweep = []
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        r = evaluate_loo(kps, fps, total, cue_frames, X, y, cue_ids, frames,
                         full_features, full_idx, threshold=thr)
        sweep.append((thr, r))
        print(f"  thr={thr:.2f}  recall={r['recall']:.1%} ({r['tp']}/{len(cue_frames)})  "
              f"fp/min={r['fp_per_min']:.2f}  lat_med={r['lat_median']:+.2f}s")

    # Reference: 84-feature production numbers from the latest live_detector run
    print("\nReference (84-feature production):")
    prod_ref = {
        0.20: ("90.9%", "10.79 fp/min"),
        0.25: ("86.4%", " 9.06 fp/min"),
        0.30: ("81.8%", " 6.47 fp/min"),
        0.35: ("77.3%", " 5.18 fp/min"),
        0.40: ("72.7%", " 4.32 fp/min"),
        0.45: ("68.2%", " 3.88 fp/min"),
        0.50: ("63.6%", " 3.02 fp/min"),
    }
    print(f"  {'thr':>5}   {'84-feat':<20}   {'792-feat':<24}")
    for thr, r in sweep:
        old = prod_ref.get(thr, ("?", "?"))
        new_str = f"{r['recall']:.1%}  {r['fp_per_min']:.2f} fp/min"
        print(f"  {thr:>4.2f}    {old[0]:<7}  {old[1]:<10}    {new_str}")

    # ── Save report ──────────────────────────────────────────────────
    lines = []
    lines.append("# RF v2 with 792 rich features — LOO results\n")
    lines.append(f"X shape: {X.shape}, y pos={y.sum()} neg={(y==0).sum()}")
    lines.append(f"\nLDA training accuracy: {lda.score(Xs, y):.3f}\n")
    lines.append("\n## Top-15 LDA discriminative features\n")
    for ix in order[:15]:
        lines.append(f"  {coef[ix]:+8.4f}   {fnames[ix]}")
    lines.append("\n## Threshold sweep — RF v2 (792 feats)\n")
    lines.append(f"  {'thr':>5}  {'recall':>7}  {'fp/min':>8}  {'lat med':>8}")
    for thr, r in sweep:
        lines.append(f"  {thr:>4.2f}   {r['recall']:>6.1%}   "
                     f"{r['fp_per_min']:>7.2f}  {r['lat_median']:+7.2f}s")
    lines.append("\n## Comparison to 84-feature production\n")
    lines.append(f"  {'thr':>5}  {'84-feat recall':<14}  {'792-feat recall':<16}  "
                 f"{'84 fp/min':<10}  {'792 fp/min':<10}")
    for thr, r in sweep:
        old_r, old_fp = prod_ref.get(thr, ("?", "?"))
        lines.append(f"  {thr:>4.2f}   {old_r:<12}    {r['recall']:.1%}            "
                     f"{old_fp:<10}  {r['fp_per_min']:.2f}")
    (OUT / "rf_v2_rich_report.txt").write_text("\n".join(lines))
    print(f"\nSaved: {OUT / 'rf_v2_rich_report.txt'}")


if __name__ == "__main__":
    main()
