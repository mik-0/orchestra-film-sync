#!/usr/bin/env python3
"""Empirical cue analysis v2: raw signals + visual thumbnail strips + manual labels.

Key improvements over v1:
- Uses RAW signal magnitudes (pixels/frame) instead of self-normalized [0,1] values
- Generates thumbnail strips around each transition for manual cue classification
- Supports manual cue labels for proper supervised analysis
- Computes discriminative power of each feature via effect size (Cohen's d)
- Optimizes weights using labeled data

Usage:
    python3 analyze_cues_v2.py                    # full analysis
    python3 analyze_cues_v2.py --labels 1,1,0,1,...  # provide manual labels
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
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

_MEDIA_DIR = os.environ.get("FIREBIRD_MEDIA_DIR")
DEFAULT_VIDEO = os.environ.get("FIREBIRD_VIDEO")
if DEFAULT_VIDEO is None and _MEDIA_DIR:
    DEFAULT_VIDEO = str(Path(_MEDIA_DIR).expanduser() / "FIREBIRD CONDUCTOR BEELD_AUDIO.mov")
TIMELINE_FILE = Path(__file__).parent / "firebird_timeline.txt"
CACHE_FILE = Path(__file__).parent / ".keypoints_cache.npz"
RESULTS_DIR = Path(__file__).parent / "analysis_results"


def parse_timeline(path: Path, fps: float = 25.0):
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        seg_id = parts[0]
        m = re.match(r"(\d+):(\d+):(\d+)", parts[1])
        if m:
            t = int(m[1]) * 60 + int(m[2]) + int(m[3]) / fps
            entries.append((seg_id, t, int(t * fps)))
    return entries


def load_keypoints():
    if not CACHE_FILE.exists():
        print("No keypoint cache found. Run analyze_cues.py first to extract.")
        sys.exit(1)
    data = np.load(CACHE_FILE)
    return data["keypoints"], float(data["fps"]), int(data["total_frames"])


def extract_thumbnails(video_path: str, timeline, fps: float, window_s=3.0):
    """Extract thumbnail strips: 7 frames spanning [-window, +window] around each transition."""
    RESULTS_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thumb_w, thumb_h = 192, 108  # 16:9 thumbnails

    n_thumbs = 7  # frames per strip
    offsets_s = np.linspace(-window_s, window_s, n_thumbs)

    for seg_id, t, frame in timeline:
        frames_to_grab = [max(0, min(total - 1, frame + int(off * fps))) for off in offsets_s]
        strip = np.zeros((thumb_h + 30, thumb_w * n_thumbs, 3), dtype=np.uint8)

        for i, f_idx in enumerate(frames_to_grab):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frm = cap.read()
            if ret:
                thumb = cv2.resize(frm, (thumb_w, thumb_h))
                x_off = i * thumb_w
                strip[:thumb_h, x_off:x_off + thumb_w] = thumb
                # Time label
                dt = offsets_s[i]
                label = f"{dt:+.1f}s"
                color = (0, 0, 255) if abs(dt) < 0.1 else (200, 200, 200)
                cv2.putText(strip, label, (x_off + 5, thumb_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Red border on center frame
        cx = (n_thumbs // 2) * thumb_w
        cv2.rectangle(strip, (cx, 0), (cx + thumb_w - 1, thumb_h - 1), (0, 0, 255), 3)

        path_out = RESULTS_DIR / f"thumb_{seg_id}.png"
        cv2.imwrite(str(path_out), strip)

    cap.release()
    print(f"Saved {len(timeline)} thumbnail strips to {RESULTS_DIR}/thumb_*.png")

    # Also make a combined image
    all_strips = []
    for seg_id, t, frame in timeline:
        img = cv2.imread(str(RESULTS_DIR / f"thumb_{seg_id}.png"))
        if img is not None:
            # Add segment label
            labeled = np.zeros((img.shape[0] + 25, img.shape[1], 3), dtype=np.uint8)
            labeled[25:] = img
            mins = int(t // 60)
            secs = t % 60
            cv2.putText(labeled, f"Seg {seg_id} @ {mins}:{secs:05.2f}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            all_strips.append(labeled)

    if all_strips:
        combined = np.vstack(all_strips)
        cv2.imwrite(str(RESULTS_DIR / "all_thumbnails.png"), combined)
        print(f"Saved combined: {RESULTS_DIR / 'all_thumbnails.png'}")


def compute_raw_features(kps: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Compute raw (non-normalized) features in physical units (pixels/frame)."""
    n = len(kps)
    l_wrist = kps[:, L_WRIST, :2]
    r_wrist = kps[:, R_WRIST, :2]
    l_shoulder = kps[:, L_SHOULDER, :2]
    r_shoulder = kps[:, R_SHOULDER, :2]
    l_elbow = kps[:, L_ELBOW, :2]
    r_elbow = kps[:, R_ELBOW, :2]

    # Velocities
    l_vel = np.diff(l_wrist, axis=0, prepend=l_wrist[:1])
    r_vel = np.diff(r_wrist, axis=0, prepend=r_wrist[:1])
    l_speed = np.linalg.norm(l_vel, axis=1)
    r_speed = np.linalg.norm(r_vel, axis=1)

    # ── Feature 1: Combined hand speed (px/frame) ──────────────────
    combined_speed = l_speed + r_speed

    # ── Feature 2: Speed surge ratio (fast EMA / slow EMA) ─────────
    speed_fast = np.zeros(n)
    speed_slow = np.zeros(n)
    for i in range(1, n):
        speed_fast[i] = speed_fast[i-1] + 0.15 * (combined_speed[i] - speed_fast[i-1])
        speed_slow[i] = speed_slow[i-1] + 0.02 * (combined_speed[i] - speed_slow[i-1])
    surge_ratio = np.where(speed_slow > 1.0,
                           np.maximum(0, (speed_fast - speed_slow) / speed_slow),
                           0.0)

    # ── Feature 3: Spread (wrist distance) and spread velocity ─────
    spread = np.linalg.norm(l_wrist - r_wrist, axis=1)
    spread_vel = np.diff(spread, prepend=spread[0])
    spread_opening = np.maximum(0, spread_vel)  # only opening, not closing

    # ── Feature 4: Hand height above shoulders (pixels) ────────────
    l_above = l_shoulder[:, 1] - l_wrist[:, 1]  # positive = above (y inverted)
    r_above = r_shoulder[:, 1] - r_wrist[:, 1]
    max_above = np.maximum(l_above, r_above)
    mean_above = (l_above + r_above) / 2

    # ── Feature 5: Upward velocity (px/frame, positive = up) ──────
    l_up = -l_vel[:, 1]  # negate because image y is down
    r_up = -r_vel[:, 1]
    combined_up = l_up + r_up
    upward_only = np.maximum(0, combined_up)

    # ── Feature 6: Post-stillness (low speed period before) ────────
    # Rolling min speed in 0.5s window
    half_sec = int(0.5 * fps)
    rolling_min_speed = np.zeros(n)
    for i in range(n):
        start = max(0, i - half_sec)
        rolling_min_speed[i] = np.min(combined_speed[start:i+1])

    # Stillness score: how low was the minimum speed in the recent past
    # compared to the overall baseline
    stillness_then_burst = np.zeros(n)
    for i in range(half_sec, n):
        recent_min = np.min(combined_speed[max(0, i - int(2*fps)):i])
        if speed_slow[i] > 1.0 and recent_min < speed_slow[i] * 0.3:
            stillness_then_burst[i] = combined_speed[i] * (1.0 - recent_min / speed_slow[i])

    # ── Feature 7: Acceleration (speed derivative) ─────────────────
    accel = np.diff(combined_speed, prepend=combined_speed[0])
    positive_accel = np.maximum(0, accel)

    # ── Feature 8: Vertical acceleration ───────────────────────────
    vy = -(l_vel[:, 1] + r_vel[:, 1])  # combined upward velocity
    vy_accel = np.diff(vy, prepend=vy[0])

    # ── Feature 9: Jerk (derivative of acceleration) ───────────────
    jerk = np.abs(np.diff(accel, prepend=accel[0]))

    # ── Feature 10: Hand reach from body center ────────────────────
    body_center = (l_shoulder + r_shoulder) / 2
    l_reach = np.linalg.norm(l_wrist - body_center, axis=1)
    r_reach = np.linalg.norm(r_wrist - body_center, axis=1)
    max_reach = np.maximum(l_reach, r_reach)
    shoulder_w = np.linalg.norm(l_shoulder - r_shoulder, axis=1)
    norm_reach = max_reach / (shoulder_w + 1e-8)

    # ── Feature 11: Elbow extension (arm straightness) ─────────────
    def elbow_angle(shoulder, elbow, wrist):
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_a = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    l_angle = elbow_angle(l_shoulder, l_elbow, l_wrist)
    r_angle = elbow_angle(r_shoulder, r_elbow, r_wrist)
    max_elbow_ext = np.maximum(l_angle, r_angle)

    # ── Feature 12: Speed asymmetry between hands ──────────────────
    speed_asymmetry = np.abs(l_speed - r_speed) / (l_speed + r_speed + 1e-8)

    # ── Feature 13: Trajectory curvature ───────────────────────────
    def curvature(vel):
        vel_norm = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8)
        return np.linalg.norm(np.diff(vel_norm, axis=0, prepend=vel_norm[:1]), axis=1)
    max_curvature = np.maximum(curvature(l_vel), curvature(r_vel))

    # ── Smoothed versions (0.5s EMA) ──────────────────────────────
    def ema(x, alpha=0.08):  # ~0.5s at 25fps
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = out[i-1] + alpha * (x[i] - out[i-1])
        return out

    return {
        "combined_speed": combined_speed,
        "combined_speed_smooth": ema(combined_speed),
        "surge_ratio": surge_ratio,
        "surge_ratio_smooth": ema(surge_ratio),
        "spread": spread,
        "spread_opening": spread_opening,
        "spread_opening_smooth": ema(spread_opening),
        "max_above_shoulders": max_above,
        "max_above_smooth": ema(np.maximum(0, max_above)),
        "mean_above_shoulders": mean_above,
        "upward_vel": upward_only,
        "upward_vel_smooth": ema(upward_only),
        "stillness_burst": stillness_then_burst,
        "stillness_burst_smooth": ema(stillness_then_burst),
        "positive_accel": positive_accel,
        "positive_accel_smooth": ema(positive_accel),
        "vy_accel": vy_accel,
        "jerk": jerk,
        "jerk_smooth": ema(jerk),
        "max_reach": max_reach,
        "norm_reach": norm_reach,
        "max_elbow_ext": max_elbow_ext,
        "speed_asymmetry": speed_asymmetry,
        "curvature": max_curvature,
        "curvature_smooth": ema(max_curvature),
        "speed_slow": speed_slow,
        "speed_fast": speed_fast,
    }


def extract_window_stats(
    features: dict[str, np.ndarray],
    timeline: list,
    fps: float,
    total: int,
    window_before_s: float = 3.0,
    window_after_s: float = 1.5,
) -> list[dict]:
    """Extract statistics from each feature in a window around each transition."""
    wb = int(window_before_s * fps)
    wa = int(window_after_s * fps)

    # Features to analyze (use smooth versions where available)
    analyze = [
        ("combined_speed_smooth", "Combined speed"),
        ("surge_ratio_smooth", "Speed surge"),
        ("spread_opening_smooth", "Spread opening"),
        ("max_above_smooth", "Hands above shoulders"),
        ("upward_vel_smooth", "Upward velocity"),
        ("stillness_burst_smooth", "Stillness-burst"),
        ("positive_accel_smooth", "Positive acceleration"),
        ("jerk_smooth", "Jerk"),
        ("norm_reach", "Normalized reach"),
        ("max_elbow_ext", "Elbow extension"),
        ("speed_asymmetry", "Speed asymmetry"),
        ("curvature_smooth", "Curvature"),
    ]

    results = []
    for seg_id, t, frame in timeline:
        start = max(0, frame - wb)
        end = min(total, frame + wa)
        # Also compute stats for subwindows
        pre_start = max(0, frame - int(1.5 * fps))
        pre_end = frame
        post_start = frame
        post_end = min(total, frame + int(1.0 * fps))
        # Immediate window: [-0.5s, +0.5s] around transition
        imm_start = max(0, frame - int(0.5 * fps))
        imm_end = min(total, frame + int(0.5 * fps))

        entry = {"seg_id": seg_id, "time": t, "frame": frame}

        for feat_key, feat_label in analyze:
            vals = features[feat_key]
            # Full window
            w = vals[start:end]
            entry[f"{feat_key}_max"] = float(np.max(w)) if len(w) > 0 else 0.0
            entry[f"{feat_key}_mean"] = float(np.mean(w)) if len(w) > 0 else 0.0
            # Peak timing relative to transition
            if len(w) > 0:
                peak_idx = start + int(np.argmax(w))
                entry[f"{feat_key}_peak_offset"] = (peak_idx - frame) / fps
            # Pre vs post
            pre = vals[pre_start:pre_end]
            post = vals[post_start:post_end]
            entry[f"{feat_key}_pre"] = float(np.mean(pre)) if len(pre) > 0 else 0.0
            entry[f"{feat_key}_post"] = float(np.mean(post)) if len(post) > 0 else 0.0
            # Immediate window (the cue moment itself)
            imm = vals[imm_start:imm_end]
            entry[f"{feat_key}_imm"] = float(np.max(imm)) if len(imm) > 0 else 0.0

        results.append(entry)

    return results


def discriminative_analysis(
    transition_data: list[dict],
    labels: np.ndarray,
    feature_keys: list[str],
    feature_labels: dict[str, str],
):
    """Compute discriminative power of each feature using Cohen's d and AUC."""
    print("\n" + "=" * 100)
    print("DISCRIMINATIVE POWER ANALYSIS (strong cues vs non-cues)")
    print("=" * 100)

    cue_idx = np.where(labels == 1)[0]
    noncue_idx = np.where(labels == 0)[0]

    if len(cue_idx) == 0 or len(noncue_idx) == 0:
        print("Need at least 1 cue and 1 non-cue for comparison.")
        return {}

    print(f"Cue transitions: {[transition_data[i]['seg_id'] for i in cue_idx]}")
    print(f"Non-cue transitions: {[transition_data[i]['seg_id'] for i in noncue_idx]}")

    # For each feature, compute multiple statistics
    stats_to_check = ["_max", "_imm", "_pre", "_post", "_mean"]
    all_results = []

    for feat_key in feature_keys:
        for stat in stats_to_check:
            full_key = f"{feat_key}{stat}"
            if full_key not in transition_data[0]:
                continue

            cue_vals = np.array([transition_data[i][full_key] for i in cue_idx])
            noncue_vals = np.array([transition_data[i][full_key] for i in noncue_idx])

            cue_mean = cue_vals.mean()
            noncue_mean = noncue_vals.mean()
            pooled_std = np.sqrt(
                ((len(cue_vals) - 1) * cue_vals.var(ddof=1) + (len(noncue_vals) - 1) * noncue_vals.var(ddof=1))
                / max(len(cue_vals) + len(noncue_vals) - 2, 1)
            )
            cohens_d = (cue_mean - noncue_mean) / (pooled_std + 1e-8)

            # Simple AUC approximation
            n_correct = 0
            n_total = 0
            for cv in cue_vals:
                for nv in noncue_vals:
                    n_total += 1
                    if cv > nv:
                        n_correct += 1
                    elif cv == nv:
                        n_correct += 0.5
            auc = n_correct / max(n_total, 1)

            # Separation
            min_cue = cue_vals.min()
            max_noncue = noncue_vals.max()
            separation = min_cue - max_noncue

            all_results.append({
                "feature": feat_key,
                "stat": stat,
                "label": feature_labels.get(feat_key, feat_key),
                "cue_mean": cue_mean,
                "noncue_mean": noncue_mean,
                "cohens_d": cohens_d,
                "auc": auc,
                "separation": separation,
                "cue_vals": cue_vals.tolist(),
                "noncue_vals": noncue_vals.tolist(),
            })

    # Sort by Cohen's d (absolute)
    all_results.sort(key=lambda x: -abs(x["cohens_d"]))

    print(f"\n{'Feature':<30} {'Stat':<6} {'Cue mean':>10} {'Non-cue':>10} "
          f"{'Cohen d':>9} {'AUC':>6} {'Sep':>8}")
    print("-" * 90)
    for r in all_results[:30]:
        print(f"{r['label']:<30} {r['stat']:<6} {r['cue_mean']:>10.2f} "
              f"{r['noncue_mean']:>10.2f} {r['cohens_d']:>9.2f} "
              f"{r['auc']:>6.2f} {r['separation']:>8.2f}")

    return all_results


def optimize_weights_supervised(
    transition_data: list[dict],
    labels: np.ndarray,
):
    """Grid search for optimal weights using the 5 original features with manual labels.

    Uses raw feature max values, properly normalized by percentile scaling.
    """
    print("\n" + "=" * 80)
    print("SUPERVISED WEIGHT OPTIMIZATION (5 original features)")
    print("=" * 80)

    # Original 5 features (raw max in window)
    feature_keys = [
        "surge_ratio_smooth",
        "spread_opening_smooth",
        "max_above_smooth",
        "upward_vel_smooth",
        "stillness_burst_smooth",
    ]
    feature_names = ["Surge", "Spread", "Height", "Upward", "Stillness"]

    n_trans = len(transition_data)
    X_raw = np.zeros((n_trans, 5))
    for i, entry in enumerate(transition_data):
        for j, fk in enumerate(feature_keys):
            X_raw[i, j] = entry[f"{fk}_max"]

    # Normalize each feature to [0, 1] using min-max across all transitions
    X = np.zeros_like(X_raw)
    for j in range(5):
        col = X_raw[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            X[:, j] = (col - lo) / (hi - lo)
        else:
            X[:, j] = 0.0

    print("\nNormalized feature matrix:")
    print(f"{'Seg':>5}", end="")
    for name in feature_names:
        print(f" {name:>10}", end="")
    print(f" {'Label':>6}")
    for i, entry in enumerate(transition_data):
        print(f"{entry['seg_id']:>5}", end="")
        for j in range(5):
            print(f" {X[i,j]:>10.3f}", end="")
        print(f" {'CUE' if labels[i] else '-':>6}")

    cue_idx = np.where(labels == 1)[0]
    noncue_idx = np.where(labels == 0)[0]

    if len(cue_idx) == 0 or len(noncue_idx) == 0:
        print("Need labeled data to optimize.")
        return

    # Grid search
    step = 0.05
    best_obj = -999
    best_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    results = []

    levels = np.arange(0, 1.001, step)
    for w0 in levels:
        for w1 in levels:
            for w2 in levels:
                for w3 in levels:
                    w4 = 1.0 - w0 - w1 - w2 - w3
                    if w4 < -0.001 or w4 > 1.001:
                        continue
                    w4 = max(0, w4)
                    w = np.array([w0, w1, w2, w3, w4])
                    scores = X @ w

                    mean_cue = scores[cue_idx].mean()
                    mean_noncue = scores[noncue_idx].mean()
                    min_cue = scores[cue_idx].min()
                    max_noncue = scores[noncue_idx].max()

                    separation = min_cue - max_noncue
                    mean_sep = mean_cue - mean_noncue

                    # Prefer clear separation, with mean separation as tiebreaker
                    obj = separation + 0.3 * mean_sep

                    results.append((obj, w.copy(), separation, mean_sep,
                                   scores[cue_idx].tolist(), scores[noncue_idx].tolist()))

    results.sort(key=lambda x: -x[0])

    print(f"\nTop 10 weight configurations:")
    for rank, (obj, w, sep, msep, cue_s, noncue_s) in enumerate(results[:10]):
        wdict = {name: f"{v:.2f}" for name, v in zip(feature_names, w)}
        print(f"  #{rank+1}: {wdict}")
        print(f"       Cue scores:     {[f'{s:.3f}' for s in cue_s]}")
        print(f"       Non-cue scores: {[f'{s:.3f}' for s in noncue_s]}")
        print(f"       Separation: {sep:.3f}, Mean sep: {msep:.3f}")

    best_w = results[0][1]
    print(f"\nBest weights: {dict(zip(feature_names, best_w.round(3)))}")

    # Also report: which feature alone best separates?
    print("\nSingle-feature discrimination:")
    for j, name in enumerate(feature_names):
        cue_mean = X[cue_idx, j].mean()
        noncue_mean = X[noncue_idx, j].mean()
        min_cue = X[cue_idx, j].min()
        max_noncue = X[noncue_idx, j].max()
        print(f"  {name:<12}: cue_mean={cue_mean:.3f}, noncue_mean={noncue_mean:.3f}, "
              f"separation={min_cue - max_noncue:.3f}")

    return dict(zip(feature_names, best_w.round(3).tolist()))


def generate_plots(
    features: dict[str, np.ndarray],
    timeline: list,
    transition_data: list[dict],
    labels: np.ndarray,
    fps: float,
    total: int,
):
    """Generate comprehensive plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    wb = int(3.0 * fps)
    wa = int(1.5 * fps)

    # ── 1. Raw signal profiles around each transition ───────────────
    plot_features = [
        ("combined_speed_smooth", "Combined speed (px/frame)"),
        ("surge_ratio_smooth", "Speed surge ratio"),
        ("spread_opening_smooth", "Spread opening (px/frame)"),
        ("max_above_smooth", "Hands above shoulders (px)"),
        ("upward_vel_smooth", "Upward velocity (px/frame)"),
        ("stillness_burst_smooth", "Stillness-burst"),
    ]
    colors_feat = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]

    fig, axes = plt.subplots(4, 4, figsize=(28, 22))
    axes_flat = axes.flatten()

    for idx, (entry, (seg_id, t, frame)) in enumerate(zip(transition_data, timeline)):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        start = max(0, frame - wb)
        end = min(total, frame + wa)
        time_axis = (np.arange(start, end) - frame) / fps

        for (fk, flabel), color in zip(plot_features, colors_feat):
            vals = features[fk][start:end]
            # Normalize to [0,1] for visualization
            vmax = vals.max() if vals.max() > 0 else 1
            ax.plot(time_axis, vals / vmax, label=flabel, color=color, linewidth=1.3, alpha=0.8)

        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        is_cue = labels[idx] == 1 if idx < len(labels) else False
        marker = "CUE" if is_cue else "no cue"
        color_title = "red" if is_cue else "gray"
        ax.set_title(f"Seg {seg_id} ({int(t//60)}:{t%60:04.1f}s) [{marker}]",
                     fontweight="bold" if is_cue else "normal", color=color_title)
        ax.set_ylim(-0.05, 1.3)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(timeline), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Raw Signal Profiles (normalized per-window for comparison)", fontsize=16)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "raw_signal_profiles.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'raw_signal_profiles.png'}")
    plt.close(fig)

    # ── 2. Feature comparison: cue vs non-cue ──────────────────────
    cue_idx = np.where(labels == 1)[0]
    noncue_idx = np.where(labels == 0)[0]

    if len(cue_idx) > 0 and len(noncue_idx) > 0:
        analyze_keys = [
            "combined_speed_smooth", "surge_ratio_smooth", "spread_opening_smooth",
            "max_above_smooth", "upward_vel_smooth", "stillness_burst_smooth",
            "positive_accel_smooth", "jerk_smooth", "curvature_smooth",
        ]
        analyze_labels = [
            "Speed", "Surge", "Spread", "Height", "Upward", "Stillness",
            "Accel", "Jerk", "Curvature",
        ]

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(analyze_keys))
        width = 0.35

        cue_maxes = []
        noncue_maxes = []
        for fk in analyze_keys:
            cv = [transition_data[i][f"{fk}_max"] for i in cue_idx]
            nv = [transition_data[i][f"{fk}_max"] for i in noncue_idx]
            cue_maxes.append(np.mean(cv))
            noncue_maxes.append(np.mean(nv))

        # Normalize to make features comparable
        max_vals = np.maximum(cue_maxes, noncue_maxes)
        max_vals = np.where(max_vals > 0, max_vals, 1)
        cue_norm = np.array(cue_maxes) / max_vals
        noncue_norm = np.array(noncue_maxes) / max_vals

        bars1 = ax.bar(x - width/2, cue_norm, width, label=f"Cue (n={len(cue_idx)})",
                       color="#e74c3c", alpha=0.8)
        bars2 = ax.bar(x + width/2, noncue_norm, width, label=f"Non-cue (n={len(noncue_idx)})",
                       color="#95a5a6", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(analyze_labels, rotation=45, ha="right")
        ax.set_ylabel("Normalized feature value")
        ax.legend()
        ax.set_title("Feature Comparison: Cue vs Non-Cue Transitions (mean of max values)")
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "cue_vs_noncue_features.png", dpi=150)
        print(f"Saved: {RESULTS_DIR / 'cue_vs_noncue_features.png'}")
        plt.close(fig)

        # ── 3. Average temporal profiles: cue vs non-cue ──────────────
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes_flat = axes.flatten()

        for idx, ((fk, flabel), color) in enumerate(zip(plot_features, colors_feat)):
            ax = axes_flat[idx]

            for group_name, group_idx, style in [
                ("Cue", cue_idx, "-"),
                ("Non-cue", noncue_idx, "--"),
            ]:
                all_windows = []
                window_len = wb + wa
                for i in group_idx:
                    _, t, frame = timeline[i]
                    start = max(0, frame - wb)
                    end = min(total, frame + wa)
                    w = features[fk][start:end]
                    if len(w) < window_len:
                        w = np.pad(w, (0, window_len - len(w)), mode="edge")
                    else:
                        w = w[:window_len]
                    all_windows.append(w)

                if all_windows:
                    mean_w = np.mean(all_windows, axis=0)
                    std_w = np.std(all_windows, axis=0) if len(all_windows) > 1 else np.zeros_like(mean_w)
                    t_axis = (np.arange(window_len) - wb) / fps

                    c = "#e74c3c" if group_name == "Cue" else "#95a5a6"
                    ax.plot(t_axis, mean_w, style, color=c, linewidth=2, label=group_name)
                    ax.fill_between(t_axis, mean_w - std_w, mean_w + std_w, color=c, alpha=0.15)

            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(flabel)
            ax.set_xlabel("Time relative to transition (s)")
            if idx == 0:
                ax.legend()

        plt.suptitle("Average Temporal Profiles: Cue vs Non-Cue", fontsize=16)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "temporal_cue_vs_noncue.png", dpi=150)
        print(f"Saved: {RESULTS_DIR / 'temporal_cue_vs_noncue.png'}")
        plt.close(fig)

    # ── 4. Full timeline with cue markers ──────────────────────────
    fig, axes = plt.subplots(7, 1, figsize=(28, 16), sharex=True)
    time_axis = np.arange(total) / fps

    for idx, ((fk, flabel), color) in enumerate(zip(plot_features, colors_feat)):
        axes[idx].plot(time_axis, features[fk], color=color, linewidth=0.5)
        axes[idx].set_ylabel(flabel, fontsize=8)
        for i, entry in enumerate(transition_data):
            c = "red" if labels[i] else "lightgray"
            lw = 2 if labels[i] else 0.5
            axes[idx].axvline(entry["time"], color=c, alpha=0.7, linewidth=lw)

    # Combined score with current weights
    score = (
        0.25 * features["surge_ratio_smooth"]
        + 0.25 * features["spread_opening_smooth"]
        + 0.20 * features["max_above_smooth"]
        + 0.15 * features["upward_vel_smooth"]
        + 0.15 * features["stillness_burst_smooth"]
    )
    axes[6].plot(time_axis, score, "k-", linewidth=0.5)
    axes[6].set_ylabel("Weighted (current)")
    for i, entry in enumerate(transition_data):
        c = "red" if labels[i] else "lightgray"
        axes[6].axvline(entry["time"], color=c, alpha=0.7, linewidth=2 if labels[i] else 0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Full Timeline: Raw Features + Cue Markers (red=cue, gray=non-cue)", fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "full_timeline_raw.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'full_timeline_raw.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        required=DEFAULT_VIDEO is None,
        help="Conductor video (or set FIREBIRD_VIDEO/FIREBIRD_MEDIA_DIR).",
    )
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated 0/1 labels for each transition (1=cue, 0=no cue)")
    args = parser.parse_args()

    kps, fps, total = load_keypoints()
    timeline = parse_timeline(TIMELINE_FILE, fps)
    n_trans = len(timeline)

    print(f"Loaded {total} frames @ {fps}fps, {n_trans} transitions")

    # Extract thumbnails for visual classification
    print("\nExtracting thumbnails...")
    extract_thumbnails(args.video, timeline, fps)

    # Compute raw features
    print("\nComputing raw features...")
    features = compute_raw_features(kps, fps)

    # Extract window stats
    print("\nExtracting window statistics...")
    transition_data = extract_window_stats(features, timeline, fps, total)

    # Labels
    if args.labels:
        labels = np.array([int(x) for x in args.labels.split(",")])
        assert len(labels) == n_trans, f"Expected {n_trans} labels, got {len(labels)}"
    else:
        # Default: use heuristic based on raw speed patterns
        # Look at acceleration into the transition and overall activity
        labels = np.zeros(n_trans, dtype=int)
        print("\nNo manual labels provided. Using speed-based heuristic for initial classification.")
        print("Re-run with --labels to use manual classification.\n")

        # Print raw data for each transition to help manual labeling
        print(f"{'Seg':>4} {'Time':>8} {'Speed':>7} {'Surge':>7} {'Spread':>7} "
              f"{'Height':>7} {'Upward':>7} {'Still':>7} {'Accel':>7} {'Jerk':>7}")
        print("-" * 80)
        for i, entry in enumerate(transition_data):
            print(
                f"{entry['seg_id']:>4} "
                f"{int(entry['time']//60):02d}:{entry['time']%60:05.2f} "
                f"{entry['combined_speed_smooth_max']:>7.1f} "
                f"{entry['surge_ratio_smooth_max']:>7.2f} "
                f"{entry['spread_opening_smooth_max']:>7.1f} "
                f"{entry['max_above_smooth_max']:>7.1f} "
                f"{entry['upward_vel_smooth_max']:>7.1f} "
                f"{entry['stillness_burst_smooth_max']:>7.1f} "
                f"{entry['positive_accel_smooth_max']:>7.1f} "
                f"{entry['jerk_smooth_max']:>7.1f} "
            )

        # Heuristic: strong surge + high speed at transition time = likely cue
        for i, entry in enumerate(transition_data):
            surge = entry["surge_ratio_smooth_max"]
            speed = entry["combined_speed_smooth_max"]
            accel = entry["positive_accel_smooth_max"]
            spread = entry["spread_opening_smooth_max"]
            # A real cue typically has multiple strong signals
            strong_signals = sum([
                surge > 0.5,
                speed > 40,
                accel > 5,
                spread > 3,
            ])
            labels[i] = 1 if strong_signals >= 3 else 0

        print(f"\nHeuristic labels: {labels.tolist()}")
        cue_segs = [transition_data[i]["seg_id"] for i in range(n_trans) if labels[i]]
        noncue_segs = [transition_data[i]["seg_id"] for i in range(n_trans) if not labels[i]]
        print(f"  Cue: {cue_segs}")
        print(f"  Non-cue: {noncue_segs}")

    # Feature keys for analysis
    feature_keys = [
        "combined_speed_smooth", "surge_ratio_smooth", "spread_opening_smooth",
        "max_above_smooth", "upward_vel_smooth", "stillness_burst_smooth",
        "positive_accel_smooth", "jerk_smooth", "norm_reach",
        "max_elbow_ext", "speed_asymmetry", "curvature_smooth",
    ]
    feature_labels = {
        "combined_speed_smooth": "Combined speed",
        "surge_ratio_smooth": "Speed surge",
        "spread_opening_smooth": "Spread opening",
        "max_above_smooth": "Hands above shoulders",
        "upward_vel_smooth": "Upward velocity",
        "stillness_burst_smooth": "Stillness-burst",
        "positive_accel_smooth": "Positive acceleration",
        "jerk_smooth": "Jerk",
        "norm_reach": "Normalized reach",
        "max_elbow_ext": "Elbow extension",
        "speed_asymmetry": "Speed asymmetry",
        "curvature_smooth": "Curvature",
    }

    # Discriminative analysis
    disc_results = discriminative_analysis(transition_data, labels, feature_keys, feature_labels)

    # Weight optimization
    opt_weights = optimize_weights_supervised(transition_data, labels)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(features, timeline, transition_data, labels, fps, total)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "transitions": transition_data,
        "labels": labels.tolist(),
        "cue_segments": [transition_data[i]["seg_id"] for i in range(n_trans) if labels[i]],
        "optimized_weights": opt_weights,
        "discrimination": disc_results[:20] if disc_results else [],
    }
    with open(RESULTS_DIR / "analysis_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'analysis_v2.json'}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review thumbnail strips in analysis_results/all_thumbnails.png")
    print("2. Classify each transition as cue (1) or non-cue (0)")
    print(f"3. Re-run: python3 analyze_cues_v2.py --labels <{n_trans} comma-separated 0/1 values>")


if __name__ == "__main__":
    main()
