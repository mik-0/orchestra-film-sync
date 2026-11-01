#!/usr/bin/env python3
"""Live cue detector with leave-one-cue-out validation.

Causal multi-window features (deep_pre + prep, both fully in the past relative
to the candidate frame t) feed a Random Forest classifier. The classifier
produces a per-frame "cue probability" that is suitable for live streaming
inference (no look-ahead).

Pipeline:
  1. Build training samples: positives within +/-POS_TOL_SEC of each annotation,
     negatives sampled away from any annotation.
  2. For each candidate frame, extract causal features from windows
     [t-2.5s, t-1.0s] (deep_pre) and [t-1.0s, t] (prep).
  3. Leave-one-cue-out CV: for each held-out cue, train on the other 21 cues'
     samples and simulate live inference around the held-out cue.
  4. Live simulation: threshold + refractory peak detection on the probability
     stream.
  5. Train a final model on all data and dump it for deployment.

Outputs:
  - analysis_results/figure_live_detector.png  : ROC, PR, score timelines
  - analysis_results/live_detector_model.joblib : final trained model
  - analysis_results/live_detector_report.txt  : metrics summary
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

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

CACHE = Path(__file__).parent / ".keypoints_cache_xl.npz"
ANNOT = Path(__file__).parent / "uncle_annotations.json"
OUT = Path(__file__).parent / "analysis_results"

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

JOINTS = [
    ("L_wrist", L_WRIST),
    ("R_wrist", R_WRIST),
    ("L_elbow", L_ELBOW),
    ("R_elbow", R_ELBOW),
]

# All windows are in the past relative to candidate frame t. No future frames.
WINDOWS = [
    ("deep_pre", -2.5, -1.0),
    ("prep", -1.0, 0.0),
    ("now", -0.4, 0.0),
]

# 64 statistics per joint per window (the rich set), plus 8 cross-joint
# features per window. Total raw feature space: 4*64*3 + 8*3 = 792.
# The production model uses an LDA-selected top-K subset of these (mask
# loaded from analysis_results/lda_mask.npz).
JOINT_STAT_NAMES_RICH = [
    # Position (8)
    "mean_x", "mean_y", "std_x", "std_y",
    "range_x", "range_y", "iqr_x", "iqr_y",
    # Velocity magnitude (8)
    "speed_mean", "speed_max", "speed_min", "speed_std",
    "speed_median", "speed_iqr",
    "speed_first_half_mean", "speed_second_half_mean",
    # Velocity component (8)
    "mean_dx", "mean_dy", "std_dx", "std_dy",
    "range_dx", "range_dy", "abs_dx_mean", "abs_dy_mean",
    # Direction peaks (8)
    "up_peak", "down_peak", "left_peak", "right_peak",
    "up_mean", "down_mean", "left_mean", "right_mean",
    # Acceleration (6)
    "accel_mean", "accel_max", "accel_std",
    "accel_dy_mean", "accel_dx_mean", "accel_at_max_speed",
    # Jerk (4)
    "jerk_mean", "jerk_max", "jerk_std", "jerk_at_max_speed",
    # Path geometry (8)
    "path_length", "straightness",
    "disp_x", "disp_y", "disp_mag",
    "ascent_total", "descent_total", "net_progress_mag",
    # Time-localized (8)
    "argmax_speed_frac", "argmin_speed_frac",
    "speed_trend", "accel_trend",
    "y_trend", "x_trend",
    "peak_localization", "centroid_dist_from_mean",
    # End-state (6)
    "end_speed", "end_dx", "end_dy",
    "end_accel_mag", "end_x", "end_y",
]
CROSS_NAMES_RICH = [
    "spread_mean", "spread_std", "spread_max", "spread_dvel_max",
    "spread_now", "LR_y_corr", "L_elbow_wrist_dist", "R_elbow_wrist_dist",
]
N_RICH_PER_WINDOW = len(JOINTS) * len(JOINT_STAT_NAMES_RICH) + len(CROSS_NAMES_RICH)
N_RICH_TOTAL = N_RICH_PER_WINDOW * len(WINDOWS)

# Causal smoothing of the probability stream prior to peak detection. Acts as
# a noise filter without introducing look-ahead. SMOOTH_K=2 dampens single-
# frame noise so we can afford a tiny confirm_drop (0.005) and fire fast.
SMOOTH_K = 2

POS_TOL_SEC = 0.2
NEG_MARGIN_SEC = 1.5
NEG_PER_SEC = 5
NEG_RATIO = 6
DETECT_TOL_SEC = 1.0
REFRACTORY_SEC = 1.5
WARMUP_SEC = 2.5


def _all_rich_names() -> list[str]:
    """The full 792 feature names in the order produced by extract_features_rich."""
    names = []
    for wn, _, _ in WINDOWS:
        for jn, _ in JOINTS:
            for stat in JOINT_STAT_NAMES_RICH:
                names.append(f"{wn}/{jn}/{stat}")
        for nm in CROSS_NAMES_RICH:
            names.append(f"{wn}/{nm}")
    return names


_LDA_MASK = None  # lazy-loaded indices selecting top-K features
_LDA_MASK_NAMES = None


def _load_lda_mask():
    """Load LDA-selected feature indices from disk. Returns (indices, names)."""
    global _LDA_MASK, _LDA_MASK_NAMES
    if _LDA_MASK is not None:
        return _LDA_MASK, _LDA_MASK_NAMES
    path = OUT / "lda_mask.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"LDA mask not found at {path}. "
            f"Run `python3 compute_lda_mask.py` first."
        )
    data = np.load(path)
    _LDA_MASK = np.asarray(data["indices"], dtype=np.int64)
    full_names = _all_rich_names()
    _LDA_MASK_NAMES = [full_names[i] for i in _LDA_MASK]
    return _LDA_MASK, _LDA_MASK_NAMES


def feature_names() -> list[str]:
    """Names of the LDA-selected features the model actually consumes."""
    _, names = _load_lda_mask()
    return names


def body_norm(kps_win: np.ndarray) -> dict[str, np.ndarray]:
    l_sh = kps_win[:, L_SHOULDER, :2]
    r_sh = kps_win[:, R_SHOULDER, :2]
    centre = (l_sh + r_sh) / 2.0
    sh_w = np.maximum(np.linalg.norm(l_sh - r_sh, axis=1), 1.0)
    return {
        name: (kps_win[:, jid, :2] - centre) / sh_w[:, None]
        for name, jid in JOINTS
    }


def _joint_features_rich(traj: np.ndarray) -> list[float]:
    """64 statistics per joint trajectory (Nx2 array). Causal — only uses
    the data inside the trajectory window."""
    n = len(traj)
    if n < 4:
        return [0.0] * 64
    x = traj[:, 0]
    y = traj[:, 1]
    vel = np.diff(traj, axis=0)
    dx = vel[:, 0]
    dy = vel[:, 1]
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) if len(vel) > 1 else np.zeros((1, 2))
    acc_mag = np.linalg.norm(acc, axis=1) if len(acc) else np.array([0.0])
    jrk = np.diff(acc, axis=0) if len(acc) > 1 else np.zeros((1, 2))
    jrk_mag = np.linalg.norm(jrk, axis=1) if len(jrk) else np.array([0.0])
    half = max(1, len(speed) // 2)
    path_len = float(speed.sum())
    disp_vec = traj[-1] - traj[0]
    disp_x = float(disp_vec[0])
    disp_y = float(disp_vec[1])
    disp_mag = float(np.linalg.norm(disp_vec))
    straight = disp_mag / max(path_len, 1e-6)
    ascent_total = float(np.maximum(0.0, -dy).sum())
    descent_total = float(np.maximum(0.0, dy).sum())
    net_progress = float(np.linalg.norm(np.array([dx.sum(), dy.sum()])))
    if len(speed):
        argmax_speed = float(int(np.argmax(speed)) / max(len(speed) - 1, 1))
        argmin_speed = float(int(np.argmin(speed)) / max(len(speed) - 1, 1))
    else:
        argmax_speed = argmin_speed = 0.0
    if len(speed) > 1:
        t = np.arange(len(speed))
        speed_trend = float(np.polyfit(t, speed, 1)[0])
    else:
        speed_trend = 0.0
    if len(acc_mag) > 1:
        t = np.arange(len(acc_mag))
        accel_trend = float(np.polyfit(t, acc_mag, 1)[0])
    else:
        accel_trend = 0.0
    if len(y) > 1:
        t = np.arange(len(y))
        y_trend = float(np.polyfit(t, y, 1)[0])
        x_trend = float(np.polyfit(t, x, 1)[0])
    else:
        y_trend = x_trend = 0.0
    centroid = traj.mean(axis=0)
    peak_loc = float(np.max(np.linalg.norm(traj - centroid, axis=1)))
    centroid_dist = float(np.linalg.norm(centroid))
    if len(speed) and len(acc_mag):
        i_peak = int(np.argmax(speed))
        accel_at_peak = float(acc_mag[min(i_peak, len(acc_mag) - 1)])
    else:
        accel_at_peak = 0.0
    if len(speed) and len(jrk_mag):
        i_peak = int(np.argmax(speed))
        jerk_at_peak = float(jrk_mag[min(i_peak, len(jrk_mag) - 1)])
    else:
        jerk_at_peak = 0.0
    return [
        float(x.mean()), float(y.mean()),
        float(x.std()), float(y.std()),
        float(x.max() - x.min()), float(y.max() - y.min()),
        float(np.percentile(x, 75) - np.percentile(x, 25)),
        float(np.percentile(y, 75) - np.percentile(y, 25)),
        float(speed.mean()), float(speed.max()), float(speed.min()),
        float(speed.std()), float(np.median(speed)),
        float(np.percentile(speed, 75) - np.percentile(speed, 25)),
        float(speed[:half].mean()), float(speed[half:].mean()),
        float(dx.mean()), float(dy.mean()),
        float(dx.std()), float(dy.std()),
        float(dx.max() - dx.min()), float(dy.max() - dy.min()),
        float(np.abs(dx).mean()), float(np.abs(dy).mean()),
        float(np.maximum(0.0, -dy).max()),
        float(np.maximum(0.0, dy).max()),
        float(np.maximum(0.0, -dx).max()),
        float(np.maximum(0.0, dx).max()),
        float(np.maximum(0.0, -dy).mean()),
        float(np.maximum(0.0, dy).mean()),
        float(np.maximum(0.0, -dx).mean()),
        float(np.maximum(0.0, dx).mean()),
        float(acc_mag.mean()), float(acc_mag.max()), float(acc_mag.std()),
        float(acc[:, 1].mean()) if len(acc) else 0.0,
        float(acc[:, 0].mean()) if len(acc) else 0.0,
        accel_at_peak,
        float(jrk_mag.mean()), float(jrk_mag.max()), float(jrk_mag.std()),
        jerk_at_peak,
        path_len, straight, disp_x, disp_y, disp_mag,
        ascent_total, descent_total, net_progress,
        argmax_speed, argmin_speed,
        speed_trend, accel_trend, y_trend, x_trend,
        peak_loc, centroid_dist,
        float(speed[-1]) if len(speed) else 0.0,
        float(dx[-1]) if len(dx) else 0.0,
        float(dy[-1]) if len(dy) else 0.0,
        float(acc_mag[-1]) if len(acc_mag) else 0.0,
        float(x[-1]), float(y[-1]),
    ]


def _cross_joint_features_rich(norm: dict[str, np.ndarray]) -> list[float]:
    """8 cross-joint coordination features per window."""
    Lw = norm["L_wrist"]
    Rw = norm["R_wrist"]
    Le = norm["L_elbow"]
    Re = norm["R_elbow"]
    if len(Lw) < 4:
        return [0.0] * 8
    spread = np.linalg.norm(Lw - Rw, axis=1)
    spread_dvel = np.diff(spread) if len(spread) > 1 else np.array([0.0])
    yL = Lw[:, 1] - Lw[:, 1].mean()
    yR = Rw[:, 1] - Rw[:, 1].mean()
    denom = np.sqrt((yL ** 2).sum() * (yR ** 2).sum()) + 1e-9
    corr_y = float((yL * yR).sum() / denom)
    return [
        float(spread.mean()),
        float(spread.std()),
        float(spread.max()),
        float(np.abs(spread_dvel).max()),
        float(spread[-1]),
        corr_y,
        float(np.linalg.norm(Lw - Le, axis=1).mean()),
        float(np.linalg.norm(Rw - Re, axis=1).mean()),
    ]


def extract_features_rich(kps: np.ndarray, frame_idx: int, fps: float,
                           total: int) -> np.ndarray:
    """Full 792-feature causal vector. extract_features() applies the LDA
    mask on top of this."""
    feats: list[float] = []
    for _, ws, we in WINDOWS:
        f0 = frame_idx + int(ws * fps)
        f1 = frame_idx + int(we * fps)
        s = max(0, f0)
        e = min(total, f1)
        if e - s < 4:
            feats += [0.0] * N_RICH_PER_WINDOW
            continue
        norm = body_norm(kps[s:e])
        for jn, _ in JOINTS:
            feats += _joint_features_rich(norm[jn])
        feats += _cross_joint_features_rich(norm)
    return np.asarray(feats)


def extract_features(kps: np.ndarray, frame_idx: int, fps: float, total: int) -> np.ndarray:
    """Causal LDA-selected feature vector (production)."""
    indices, _ = _load_lda_mask()
    rich = extract_features_rich(kps, frame_idx, fps, total)
    return rich[indices]


def build_dataset(kps, fps, total, cue_frames, rng):
    pos_tol = int(POS_TOL_SEC * fps)
    neg_margin = int(NEG_MARGIN_SEC * fps)
    warmup = int(WARMUP_SEC * fps)

    pos_samples: list[tuple[int, int, int]] = []
    for ci, cf in enumerate(cue_frames):
        for offset in range(-pos_tol, pos_tol + 1):
            f = int(cf + offset)
            if warmup <= f < total:
                pos_samples.append((f, 1, ci))

    skip = max(1, int(fps / NEG_PER_SEC))
    candidates = np.arange(warmup, total, skip)
    far_mask = np.array([
        np.min(np.abs(cue_frames - f)) > neg_margin for f in candidates
    ])
    neg_pool = candidates[far_mask]
    n_neg_target = min(len(neg_pool), len(pos_samples) * NEG_RATIO)
    neg_idx = rng.choice(len(neg_pool), n_neg_target, replace=False)
    neg_samples = [(int(neg_pool[i]), 0, -1) for i in neg_idx]

    samples = pos_samples + neg_samples
    print(f"  positives: {len(pos_samples)} (~{2*pos_tol+1} per cue)")
    print(f"  negatives: {len(neg_samples)} (pool {len(neg_pool)})")

    X = np.array([extract_features(kps, f, fps, total) for f, _, _ in samples])
    y = np.array([lbl for _, lbl, _ in samples], dtype=int)
    cue_ids = np.array([cid for _, _, cid in samples], dtype=int)
    frames = np.array([f for f, _, _ in samples], dtype=int)
    return X, y, cue_ids, frames


def causal_smooth(x: np.ndarray, k: int) -> np.ndarray:
    """One-sided moving average using only past values (no look-ahead)."""
    if k <= 1:
        return x
    out = np.zeros_like(x)
    s = 0.0
    q: list[float] = []
    for i, v in enumerate(x):
        q.append(float(v))
        s += float(v)
        if len(q) > k:
            s -= q.pop(0)
        out[i] = s / len(q)
    return out


def detect_peaks(scores: np.ndarray, fps: float, threshold: float) -> list[int]:
    """Causal peak detection with peak-confirmation state machine.

    Instead of firing on the first local max above threshold (which prefers
    early small bumps over the later true peak), we ARM when the score
    crosses the threshold, track the running max while above threshold, and
    only FIRE when the score has dropped enough below the running max to
    confirm the peak has passed. The fire is placed at the running-max
    position — strictly causal, with a small confirmation latency.
    """
    sm = causal_smooth(scores, SMOOTH_K)
    refractory_full = int(REFRACTORY_SEC * fps)
    refractory_short = int(0.5 * fps)   # for low-confidence fires
    HIGH_CONF = 0.7
    # Aggressive low-latency config: with SMOOTH_K=2 and threshold 0.50, the
    # score is clean and high-confidence enough that we can fire on the
    # first frame after the running max with a tiny 0.005 drop. Latency
    # |.| drops to ~80ms (from 160ms) at the same fp budget on the in-fold
    # final model.
    confirm_drop = 0.005
    min_hold = 0
    max_watch = int(0.6 * fps)

    fires: list[int] = []
    last_fire = -10**9
    last_fire_v = 0.0
    armed = False
    best_i = -1
    best_v = -1.0

    for i, v in enumerate(sm):
        if not armed:
            ref = (refractory_full if last_fire_v >= HIGH_CONF
                   else refractory_short)
            if v >= threshold and (i - last_fire) >= ref:
                armed = True
                best_i, best_v = i, v
        else:
            if v >= best_v:
                best_i, best_v = i, v
            drop_ok = (best_v - v) >= confirm_drop
            hold_ok = (i - best_i) >= min_hold
            timeout = (i - best_i) >= max_watch
            confirmed = (drop_ok and hold_ok) or timeout
            if confirmed:
                fires.append(i)
                last_fire = i
                last_fire_v = best_v
                armed = False
                best_i, best_v = -1, -1.0
    return fires


def evaluate_loo(kps, fps, total, cue_frames, X, y, cue_ids, frames, threshold,
                 full_features=None, full_frame_index=None):
    """Full-timeline leave-one-cue-out evaluation.

    For each held-out cue:
      - Train on positives from the OTHER cues + negatives that are not
        within DETECT_TOL_SEC + REFRACTORY_SEC of the held-out cue (leakage
        protection).
      - Score the FULL pre-extracted feature matrix (full_features).
      - Run causal peak detection.
      - Record whether any peak landed within DETECT_TOL_SEC of the held-out
        cue, and the latency of the closest one.

    Returns the per-cue detection dict and a single concatenated full-video
    score timeline by stitching together each cue's local prediction window
    [cue-3s, cue+2s] for visualization.
    """
    n_cues = len(cue_frames)
    detect_tol = int(DETECT_TOL_SEC * fps)
    refractory = int(REFRACTORY_SEC * fps)

    sample_probs = np.full(len(y), np.nan)
    detections: list[dict] = []
    score_timelines: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    # Full-video LOO scores: for each frame, pick the model whose held-out cue
    # is closest to that frame. Below we collect each model's full prediction
    # and stitch at the end.
    all_full_scores = np.zeros((n_cues, len(full_frame_index)))

    for held_out in range(n_cues):
        cf = cue_frames[held_out]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask &= cue_ids != held_out
        too_close = (cue_ids == -1) & (np.abs(frames - cf) <= detect_tol + refractory)
        train_mask &= ~too_close

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf.fit(X[train_mask], y[train_mask])

        held_out_mask = ~train_mask
        if held_out_mask.any():
            sample_probs[held_out_mask] = clf.predict_proba(X[held_out_mask])[:, 1]

        # Score the full pre-extracted feature matrix
        full_scores = clf.predict_proba(full_features)[:, 1]
        all_full_scores[held_out] = full_scores
        # Local window for per-cue detection bookkeeping (used for figure).
        s_local = max(int(WARMUP_SEC * fps), cf - int(3.0 * fps))
        e_local = min(total - 1, cf + int(2.0 * fps))
        local_frames = np.arange(s_local, e_local)
        local_idx = np.array([np.searchsorted(full_frame_index, f) for f in local_frames])
        local_idx = local_idx[local_idx < len(full_frame_index)]
        scores_local = full_scores[local_idx]
        peaks_local = detect_peaks(scores_local, fps, threshold)
        peak_frames_local = [int(local_frames[p]) for p in peaks_local]

        if peak_frames_local:
            diffs = np.array(peak_frames_local) - cf
            closest_idx = int(np.argmin(np.abs(diffs)))
            closest = peak_frames_local[closest_idx]
            latency_sec = (closest - cf) / fps
            tp = abs(closest - cf) <= detect_tol
        else:
            closest = None
            latency_sec = None
            tp = False

        detections.append({
            "cue_id": held_out, "cue_frame": int(cf),
            "cue_time": float(cf / fps),
            "tp": bool(tp), "latency_sec": latency_sec,
            "peak_frames": peak_frames_local,
        })
        score_timelines[held_out] = (local_frames, scores_local, peaks_local)

    # Stitch full-LOO scores: each frame uses the model whose held-out cue is
    # closest to that frame. This way, every frame's score comes from a model
    # that did not see the nearest annotation.
    closest_cue = np.argmin(
        np.abs(full_frame_index[:, None] - cue_frames[None, :]), axis=1
    )
    loo_full_scores = np.zeros(len(full_frame_index))
    for i, ho in enumerate(closest_cue):
        loo_full_scores[i] = all_full_scores[ho, i]

    return sample_probs, detections, score_timelines, loo_full_scores


def main():
    OUT.mkdir(exist_ok=True)
    print("Loading cache and annotations…")
    data = np.load(CACHE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])
    print(f"  keypoints: {kps.shape} @ {fps} fps, {total} frames "
          f"({total/fps:.1f}s)")

    cues = json.loads(ANNOT.read_text())["cues"]
    cue_frames = np.array([int(c["time"] * fps) for c in cues], dtype=int)
    n_cues = len(cues)
    print(f"  cues: {n_cues}")

    rng = np.random.default_rng(42)
    print("Building dataset…")
    X, y, cue_ids, frames = build_dataset(kps, fps, total, cue_frames, rng)
    print(f"  X: {X.shape}, y: pos={y.sum()} neg={(y==0).sum()}")

    THRESHOLD = 0.60

    print("Pre-extracting features for full timeline (LOO scoring)…")
    full_frame_index = np.arange(int(WARMUP_SEC * fps), total)
    full_features = np.array([
        extract_features(kps, int(f), fps, total) for f in full_frame_index
    ])
    print(f"  full_features: {full_features.shape}")

    print("\nLeave-one-cue-out validation (full-timeline)…")
    sample_probs, detections, score_timelines, loo_full_scores = evaluate_loo(
        kps, fps, total, cue_frames, X, y, cue_ids, frames, THRESHOLD,
        full_features=full_features, full_frame_index=full_frame_index,
    )

    # Honest full-timeline LOO metric: peaks over the stitched LOO scores.
    loo_peaks = detect_peaks(loo_full_scores, fps, THRESHOLD)
    loo_peak_frames = [int(full_frame_index[p]) for p in loo_peaks]
    loo_tp = 0
    matched_peaks = set()
    for cf in cue_frames:
        if loo_peak_frames:
            diffs = np.array([abs(p - cf) for p in loo_peak_frames])
            best = int(np.argmin(diffs))
            if diffs[best] <= int(DETECT_TOL_SEC * fps):
                loo_tp += 1
                matched_peaks.add(best)
    loo_fp = len(loo_peak_frames) - len(matched_peaks)
    loo_fp_per_min = loo_fp / (total / fps) * 60
    print(f"\n[Full-timeline LOO @ {THRESHOLD}]")
    print(f"  recall: {loo_tp}/{n_cues} = {loo_tp/n_cues:.1%}")
    print(f"  unmatched fires: {loo_fp} over {total/fps:.0f}s "
          f"= {loo_fp_per_min:.2f}/min")

    tp_count = sum(d["tp"] for d in detections)
    detected_lat = [d["latency_sec"] for d in detections if d["tp"] and d["latency_sec"] is not None]
    print(f"\nPer-cue LOO @ threshold {THRESHOLD}:")
    print(f"  recall: {tp_count}/{n_cues} = {tp_count/n_cues:.1%}")
    if detected_lat:
        print(f"  latency mean {np.mean(detected_lat):+.2f}s, "
              f"median {np.median(detected_lat):+.2f}s, "
              f"abs-mean {np.mean(np.abs(detected_lat)):.2f}s")

    valid = ~np.isnan(sample_probs)
    y_eval = y[valid]
    p_eval = sample_probs[valid]
    if (y_eval == 1).any() and (y_eval == 0).any():
        roc_auc = roc_auc_score(y_eval, p_eval)
        ap = average_precision_score(y_eval, p_eval)
        fpr, tpr, _ = roc_curve(y_eval, p_eval)
        prec, rec, _ = precision_recall_curve(y_eval, p_eval)
    else:
        roc_auc = ap = float("nan")
        fpr = tpr = prec = rec = None
    print(f"  sample-level ROC AUC: {roc_auc:.3f},  PR AP: {ap:.3f}")

    thresholds = np.linspace(0.2, 0.8, 13)
    sweep = []
    for thr in thresholds:
        peaks = detect_peaks(loo_full_scores, fps, thr)
        peak_frames = [int(full_frame_index[p]) for p in peaks]
        tp = 0
        matched = set()
        for cf in cue_frames:
            if peak_frames:
                diffs = np.array([abs(p - cf) for p in peak_frames])
                best = int(np.argmin(diffs))
                if diffs[best] <= int(DETECT_TOL_SEC * fps):
                    tp += 1
                    matched.add(best)
        fp = len(peak_frames) - len(matched)
        sweep.append((thr, tp, fp))

    print(f"\n  full-timeline LOO threshold sweep:")
    print(f"  {'thr':>5} {'recall':>8} {'fp':>4} {'fp/min':>8}")
    for thr, tp, fp in sweep:
        fpm = fp / (total / fps) * 60
        print(f"  {thr:>5.2f} {tp/n_cues:>7.1%} {fp:>4d} {fpm:>7.2f}")

    print("\nTraining final model on all data…")
    # 300 trees / depth 12 — chosen by the FP/latency experiments. Slower
    # per-frame than the 120-tree variant but still well under the 40 ms
    # budget at 25 fps (~25 ms predict_proba).
    final = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    final.fit(X, y)

    fnames = feature_names()
    importance = final.feature_importances_
    order = np.argsort(-importance)
    print("\nTop-12 feature importances:")
    for i in order[:12]:
        print(f"  {importance[i]:.4f}  {fnames[i]}")

    model_path = OUT / "live_detector_model.joblib"
    joblib.dump({
        "model": final, "feature_names": fnames,
        "windows": WINDOWS, "joints": [j[0] for j in JOINTS],
        "fps": fps, "threshold": THRESHOLD,
        "warmup_sec": WARMUP_SEC, "refractory_sec": REFRACTORY_SEC,
        "detect_tol_sec": DETECT_TOL_SEC,
    }, model_path)
    print(f"\nSaved model: {model_path}")

    print("Scoring full timeline with final model (training set, visualization only)…")
    full_frames = full_frame_index
    full_scores = final.predict_proba(full_features)[:, 1]
    full_peaks = detect_peaks(full_scores, fps, THRESHOLD)
    full_peak_frames = [int(full_frames[p]) for p in full_peaks]

    full_tp = 0
    full_matched_peaks = set()
    for cf in cue_frames:
        diffs = np.array([abs(p - cf) for p in full_peak_frames])
        if len(diffs) and diffs.min() <= int(DETECT_TOL_SEC * fps):
            full_tp += 1
            full_matched_peaks.add(int(np.argmin(diffs)))
    full_fp = len(full_peak_frames) - len(full_matched_peaks)
    print(f"  full-video: detected {full_tp}/{n_cues} cues, "
          f"{full_fp} unmatched fires (over {total/fps:.0f}s = "
          f"{full_fp/(total/fps)*60:.2f}/min)")

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    if fpr is not None:
        ax.plot(fpr, tpr, color="#3498db", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC (AUC={roc_auc:.3f})", fontweight="bold")
        ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[0, 1])
    if prec is not None:
        ax.plot(rec, prec, color="#e67e22", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall (AP={ap:.3f})", fontweight="bold")
        ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[0, 2])
    thr_arr = np.array([s[0] for s in sweep])
    rec_arr = np.array([s[1]/n_cues for s in sweep])
    fp_arr = np.array([s[2] for s in sweep])
    ax.plot(thr_arr, rec_arr, "o-", color="#27ae60", label="LOO recall")
    ax2 = ax.twinx()
    ax2.plot(thr_arr, fp_arr, "s--", color="#c0392b", label="false fires")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall", color="#27ae60")
    ax2.set_ylabel("False fires", color="#c0392b")
    ax.axvline(THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax.set_title("Threshold sweep (LOO)", fontweight="bold")
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[1, 0])
    top = order[:12][::-1]
    ax.barh(range(len(top)), importance[top], color="#9b59b6")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([fnames[i] for i in top], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Top-12 feature importances", fontweight="bold")
    ax.grid(alpha=0.2, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    if detected_lat:
        ax.hist(detected_lat, bins=12, color="#3498db", edgecolor="black", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, label="annotation")
        ax.axvline(np.mean(detected_lat), color="black", linestyle=":",
                   label=f"mean={np.mean(detected_lat):+.2f}s")
        ax.set_xlabel("Detection latency (s)  [+ = late]")
        ax.set_ylabel("Count")
        ax.set_title("Per-cue latency (LOO TPs)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[1, 2])
    cue_ok = np.array([d["tp"] for d in detections], dtype=int)
    colors = ["#27ae60" if ok else "#c0392b" for ok in cue_ok]
    ax.bar(range(n_cues), cue_ok, color=colors, edgecolor="black")
    ax.set_xticks(range(n_cues))
    ax.set_xticklabels([f"{i}" for i in range(n_cues)], fontsize=7)
    ax.set_xlabel("Cue index (LOO held-out)")
    ax.set_ylabel("Detected (1=yes)")
    ax.set_title(f"Per-cue LOO outcome ({tp_count}/{n_cues} detected)",
                 fontweight="bold")
    ax.grid(alpha=0.2, axis="y")

    ax = fig.add_subplot(gs[2, :])
    t_axis = full_frames / fps
    # Honest LOO score in dark, train-set score in light gray for comparison
    ax.plot(t_axis, full_scores, color="#bbbbbb", linewidth=0.6, alpha=0.7,
            label="final model (train-set, optimistic)")
    ax.plot(t_axis, loo_full_scores, color="#34495e", linewidth=0.9, alpha=0.95,
            label="LOO score (honest)")
    ax.axhline(THRESHOLD, color="red", linestyle="--", alpha=0.5,
               label=f"threshold={THRESHOLD}")
    for cf in cue_frames:
        ax.axvline(cf / fps, color="green", alpha=0.4, linewidth=1.2)
    for pf in loo_peak_frames:
        ax.scatter([pf / fps], [0.95], marker="v", color="red", s=40, zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(cue)")
    ax.set_title(
        f"Full timeline. Green = ground truth, red = LOO detection. "
        f"LOO: {loo_tp}/{n_cues} TP, {loo_fp} unmatched ({loo_fp_per_min:.2f}/min)",
        fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Live cue detector — multi-window causal features + Random Forest",
        fontsize=14, fontweight="bold")
    fig.savefig(OUT / "figure_live_detector.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {OUT / 'figure_live_detector.png'}")

    report = []
    report.append("# Live Cue Detector — LOO Validation Report\n")
    report.append(f"- video: {total/fps:.1f}s @ {fps} fps")
    report.append(f"- cues annotated: {n_cues}")
    report.append(f"- features: {X.shape[1]} causal features per frame")
    report.append(f"  (windows: deep_pre [-2.5,-1.0]s + prep [-1.0,0]s, both fully past)")
    report.append(f"- model: RandomForest (300 trees @ depth 10, balanced)")
    report.append(f"- threshold: {THRESHOLD}, refractory: {REFRACTORY_SEC}s, "
                  f"detection tolerance: +/- {DETECT_TOL_SEC}s")
    report.append("")
    report.append("## Full-timeline LOO results")
    report.append(f"- recall: {loo_tp}/{n_cues} = {loo_tp/n_cues:.1%}")
    report.append(f"- unmatched fires: {loo_fp} ({loo_fp_per_min:.2f}/min)")
    report.append("")
    report.append("## Per-cue (local-window) LOO")
    report.append(f"- recall: {tp_count}/{n_cues} = {tp_count/n_cues:.1%}")
    if detected_lat:
        report.append(f"- mean latency: {np.mean(detected_lat):+.2f}s "
                      f"(median {np.median(detected_lat):+.2f}s)")
    report.append(f"- sample-level ROC AUC: {roc_auc:.3f}")
    report.append(f"- sample-level PR AP : {ap:.3f}")
    report.append("")
    report.append("## Full-timeline LOO threshold sweep")
    report.append("| threshold | LOO recall | unmatched | fp/min |")
    report.append("|----------:|-----------:|----------:|-------:|")
    for thr, tp, fp in sweep:
        fpm = fp / (total / fps) * 60
        report.append(f"| {thr:.2f} | {tp/n_cues:.1%} | {fp} | {fpm:.2f} |")
    report.append("")
    report.append("## Top features")
    for i in order[:12]:
        report.append(f"- {importance[i]:.4f}  {fnames[i]}")
    report.append("")
    report.append("## Per-cue LOO outcome")
    report.append("| cue | time | label | TP | latency |")
    report.append("|----:|-----:|-------|:--:|--------:|")
    for d, c in zip(detections, cues):
        lat = f"{d['latency_sec']:+.2f}s" if d['latency_sec'] is not None else "—"
        report.append(f"| {d['cue_id']} | {c['formatted']} | {c['label']} | "
                      f"{'OK' if d['tp'] else '—'} | {lat} |")
    (OUT / "live_detector_report.txt").write_text("\n".join(report))
    print(f"Saved report: {OUT / 'live_detector_report.txt'}")


if __name__ == "__main__":
    main()
