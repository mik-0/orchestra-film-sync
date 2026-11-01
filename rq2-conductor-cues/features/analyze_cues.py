#!/usr/bin/env python3
"""Empirical analysis of conductor cue signals around Firebird timeline transitions.

Extracts YOLO pose keypoints for the full video, computes the 5 cue-detector
signals frame-by-frame, and analyzes their behavior in windows around each
known transition timestamp. Produces:
  1. Per-transition signal profile plots
  2. Correlation / importance analysis
  3. Optimized weights via logistic regression or grid search
  4. Summary statistics table

Usage:
    python3 analyze_cues.py                           # full pipeline
    python3 analyze_cues.py --skip-extraction         # reuse cached keypoints
    python3 analyze_cues.py --process-every 2         # faster extraction
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


# ── Parse timeline ───────────────────────────────────────────────────────────

def parse_timeline(path: Path, fps: float = 25.0) -> list[tuple[str, float, int]]:
    """Returns list of (segment_id, time_seconds, frame_number)."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        seg_id = parts[0]
        tc = parts[1]  # MM:SS:FF
        m = re.match(r"(\d+):(\d+):(\d+)", tc)
        if m:
            mins, secs, frames = int(m[1]), int(m[2]), int(m[3])
            t = mins * 60 + secs + frames / fps
            frame = int(t * fps)
            entries.append((seg_id, t, frame))
    return entries


# ── Extract keypoints ────────────────────────────────────────────────────────

def extract_keypoints(
    video_path: str,
    process_every: int = 1,
    imgsz: int = 640,
) -> tuple[np.ndarray, float, int]:
    """Run YOLO pose on every Nth frame, return (keypoints_array, fps, total_frames).

    keypoints_array shape: (total_frames, 17, 3) -- x, y, conf per joint.
    Frames not processed get interpolated from neighbors.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached keypoints from {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        return data["keypoints"], float(data["fps"]), int(data["total_frames"])

    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")

    all_kps = np.zeros((total, 17, 3), dtype=np.float32)
    valid_mask = np.zeros(total, dtype=bool)

    t0 = time.time()
    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % process_every != 0:
            continue
        results = model(frame, imgsz=imgsz, verbose=False)
        if (results and results[0].keypoints is not None
                and len(results[0].keypoints.data) > 0
                and results[0].boxes is not None):
            kps_data = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(areas.argmax())
            all_kps[idx] = kps_data[best]
            valid_mask[idx] = True

        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (total - idx) / rate
            print(f"  Frame {idx}/{total} ({100*idx/total:.0f}%) - {rate:.0f} fps - ETA {eta:.0f}s")

    cap.release()
    elapsed = time.time() - t0
    print(f"Extraction done in {elapsed:.1f}s ({total/elapsed:.0f} fps)")

    # Interpolate missing frames
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 1:
        for joint in range(17):
            for dim in range(3):
                vals = all_kps[valid_idx, joint, dim]
                all_kps[:, joint, dim] = np.interp(
                    np.arange(total), valid_idx, vals
                )

    np.savez_compressed(CACHE_FILE, keypoints=all_kps, fps=fps, total_frames=total)
    print(f"Cached to {CACHE_FILE}")
    return all_kps, fps, total


# ── Compute signals ──────────────────────────────────────────────────────────

def compute_signals(kps: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Compute all raw and normalized signals frame-by-frame.

    Returns dict of 1D arrays, length = number of frames.
    """
    n = len(kps)

    # Raw measurements
    l_wrist = kps[:, L_WRIST, :2]
    r_wrist = kps[:, R_WRIST, :2]
    l_shoulder = kps[:, L_SHOULDER, :2]
    r_shoulder = kps[:, R_SHOULDER, :2]

    # Frame-to-frame velocities
    l_vel = np.diff(l_wrist, axis=0, prepend=l_wrist[:1])
    r_vel = np.diff(r_wrist, axis=0, prepend=r_wrist[:1])
    l_speed = np.linalg.norm(l_vel, axis=1)
    r_speed = np.linalg.norm(r_vel, axis=1)
    combined_speed = l_speed + r_speed

    l_dy = l_vel[:, 1]
    r_dy = r_vel[:, 1]

    # Wrist spread
    spread = np.linalg.norm(l_wrist - r_wrist, axis=1)
    spread_vel = np.diff(spread, prepend=spread[0])

    # Shoulder width for normalization
    shoulder_w = np.linalg.norm(l_shoulder - r_shoulder, axis=1)

    # Hand height above shoulders (positive = above, image y inverted)
    l_height = l_shoulder[:, 1] - l_wrist[:, 1]
    r_height = r_shoulder[:, 1] - r_wrist[:, 1]
    max_height = np.maximum(l_height, r_height)

    # Upward motion (negative dy = up in image coords)
    combined_upward = np.maximum(0, -(l_dy + r_dy))

    # ── Signal 1: Speed surge (fast EMA vs slow EMA) ────────────────
    speed_fast = np.zeros(n)
    speed_slow = np.zeros(n)
    surge = np.zeros(n)
    for i in range(1, n):
        speed_fast[i] = speed_fast[i-1] + 0.15 * (combined_speed[i] - speed_fast[i-1])
        speed_slow[i] = speed_slow[i-1] + 0.02 * (combined_speed[i] - speed_slow[i-1])
        if speed_slow[i] > 1.0:
            surge[i] = max(0, (speed_fast[i] - speed_slow[i]) / speed_slow[i])
    n_surge = np.minimum(surge / 1.5, 1.0)

    # ── Signal 2: Spread velocity (opening) ─────────────────────────
    opening = np.maximum(0, spread_vel)
    ema_spread = np.zeros(n)
    max_spread = np.full(n, 1e-6)
    for i in range(1, n):
        ema_spread[i] = ema_spread[i-1] + 0.12 * (opening[i] - ema_spread[i-1])
        max_spread[i] = max(max_spread[i-1] * 0.999, ema_spread[i])
    n_spread = ema_spread / max_spread

    # ── Signal 3: Hands above shoulders ─────────────────────────────
    above = np.maximum(0, max_height)
    ema_height = np.zeros(n)
    for i in range(1, n):
        ema_height[i] = ema_height[i-1] + 0.10 * (above[i] - ema_height[i-1])
    height_ref = np.maximum(shoulder_w * 0.25, 1.0)
    n_height = np.minimum(ema_height / height_ref, 1.0)

    # ── Signal 4: Upward motion ─────────────────────────────────────
    ema_upward = np.zeros(n)
    max_upward = np.full(n, 1e-6)
    for i in range(1, n):
        ema_upward[i] = ema_upward[i-1] + 0.10 * (combined_upward[i] - ema_upward[i-1])
        max_upward[i] = max(max_upward[i-1] * 0.999, ema_upward[i])
    n_upward = ema_upward / max_upward

    # ── Signal 5: Post-stillness coiled spring ──────────────────────
    coiled = np.zeros(n)
    ema_coil = np.zeros(n)
    max_coil = np.full(n, 1e-6)
    for i in range(1, n):
        if speed_slow[i] > 1.0:
            if combined_speed[i] < speed_slow[i] * 0.5:
                coiled[i] = min(coiled[i-1] + 0.04, 1.0)
            else:
                release = coiled[i-1] * surge[i]
                ema_coil[i] = ema_coil[i-1] + 0.15 * (release - ema_coil[i-1])
                max_coil[i] = max(max_coil[i-1] * 0.999, ema_coil[i])
                coiled[i] = coiled[i-1] * 0.92
        else:
            coiled[i] = coiled[i-1]
            ema_coil[i] = ema_coil[i-1]
            max_coil[i] = max_coil[i-1]
    n_coil = ema_coil / max_coil

    # ── Peak-tracked versions (short-window max with 0.90 decay) ────
    def peak_track(signal, decay=0.90):
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = max(out[i-1] * decay, signal[i])
        return out

    p_surge = peak_track(n_surge)
    p_spread = peak_track(n_spread)
    p_height = peak_track(n_height)
    p_upward = peak_track(n_upward)
    p_coil = peak_track(n_coil)

    return {
        # Raw
        "combined_speed": combined_speed,
        "spread": spread,
        "spread_vel": spread_vel,
        "max_height": max_height,
        "combined_upward": combined_upward,
        "l_speed": l_speed,
        "r_speed": r_speed,
        "l_dy": l_dy,
        "r_dy": r_dy,
        "shoulder_w": shoulder_w,
        "coiled": coiled,
        # Normalized (before peak tracking)
        "n_surge": n_surge,
        "n_spread": n_spread,
        "n_height": n_height,
        "n_upward": n_upward,
        "n_coil": n_coil,
        # Peak-tracked (what the detector actually uses)
        "p_surge": p_surge,
        "p_spread": p_spread,
        "p_height": p_height,
        "p_upward": p_upward,
        "p_coil": p_coil,
        # EMA internals
        "speed_fast": speed_fast,
        "speed_slow": speed_slow,
        "surge_raw": surge,
    }


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_transitions(
    signals: dict[str, np.ndarray],
    timeline: list[tuple[str, float, int]],
    fps: float,
    total_frames: int,
    window_before_s: float = 4.0,
    window_after_s: float = 2.0,
) -> dict:
    """Extract signal windows around each transition and classify cue strength."""
    wb = int(window_before_s * fps)
    wa = int(window_after_s * fps)

    peak_signals = ["p_surge", "p_spread", "p_height", "p_upward", "p_coil"]
    raw_signals = ["combined_speed", "spread", "spread_vel", "max_height", "combined_upward"]

    results = []
    for seg_id, t, frame in timeline:
        start = max(0, frame - wb)
        end = min(total_frames, frame + wa)
        window_frames = np.arange(start, end)
        window_times = window_frames / fps

        entry = {
            "seg_id": seg_id,
            "time": t,
            "frame": frame,
            "window_start": start,
            "window_end": end,
        }

        # Extract peak signal values in the window
        for sig_name in peak_signals:
            window_vals = signals[sig_name][start:end]
            entry[f"{sig_name}_max"] = float(np.max(window_vals))
            entry[f"{sig_name}_mean"] = float(np.mean(window_vals))
            # Peak value relative to the transition time
            peak_frame = start + int(np.argmax(window_vals))
            entry[f"{sig_name}_peak_offset_s"] = (peak_frame - frame) / fps

        # Compute combined score at each frame using current weights
        score_current = (
            0.25 * signals["p_surge"][start:end]
            + 0.25 * signals["p_spread"][start:end]
            + 0.20 * signals["p_height"][start:end]
            + 0.15 * signals["p_upward"][start:end]
            + 0.15 * signals["p_coil"][start:end]
        )
        entry["max_score_current"] = float(np.max(score_current))

        # Raw speed stats in the window
        speed_window = signals["combined_speed"][start:end]
        # Look at speed in 1s before vs 1s after transition
        pre_start = max(0, frame - int(fps)) - start
        pre_end = frame - start
        post_start = frame - start
        post_end = min(len(speed_window), frame - start + int(fps))
        if pre_start >= 0 and pre_end > pre_start:
            entry["speed_pre_1s"] = float(np.mean(speed_window[max(0,pre_start):max(1,pre_end)]))
        else:
            entry["speed_pre_1s"] = 0.0
        if post_end > post_start:
            entry["speed_post_1s"] = float(np.mean(speed_window[post_start:post_end]))
        else:
            entry["speed_post_1s"] = 0.0

        entry["speed_ratio"] = (
            entry["speed_post_1s"] / max(entry["speed_pre_1s"], 0.1)
        )

        results.append(entry)

    return results


def classify_cues(transition_data: list[dict]) -> list[dict]:
    """Heuristic classification of transitions into strong/weak/no cue.

    We look at combined score and speed characteristics.
    """
    for entry in transition_data:
        score = entry["max_score_current"]
        speed_ratio = entry["speed_ratio"]

        # Strong cue: high score AND noticeable speed change
        if score >= 0.45 and speed_ratio >= 1.3:
            entry["cue_class"] = "strong"
        elif score >= 0.30:
            entry["cue_class"] = "weak"
        else:
            entry["cue_class"] = "none"

    return transition_data


def print_summary(transition_data: list[dict]):
    """Print a formatted table of transition analysis."""
    print("\n" + "=" * 120)
    print("TRANSITION ANALYSIS SUMMARY")
    print("=" * 120)
    print(f"{'Seg':>4} {'Time':>8} {'Class':>7} {'Score':>6} "
          f"{'Surge':>6} {'Spread':>7} {'Height':>7} {'Upward':>7} {'Coil':>6} "
          f"{'SpdPre':>7} {'SpdPost':>8} {'Ratio':>6}")
    print("-" * 120)

    for e in transition_data:
        mins = int(e["time"] // 60)
        secs = e["time"] % 60
        cls_marker = {"strong": "***", "weak": " * ", "none": "   "}[e["cue_class"]]
        print(
            f"{e['seg_id']:>4} {mins:02d}:{secs:05.2f} {cls_marker:>7} "
            f"{e['max_score_current']:>6.3f} "
            f"{e['p_surge_max']:>6.3f} {e['p_spread_max']:>7.3f} "
            f"{e['p_height_max']:>7.3f} {e['p_upward_max']:>7.3f} "
            f"{e['p_coil_max']:>6.3f} "
            f"{e['speed_pre_1s']:>7.1f} {e['speed_post_1s']:>8.1f} "
            f"{e['speed_ratio']:>6.2f}"
        )
    print("=" * 120)

    strong = [e for e in transition_data if e["cue_class"] == "strong"]
    weak = [e for e in transition_data if e["cue_class"] == "weak"]
    none_ = [e for e in transition_data if e["cue_class"] == "none"]
    print(f"\nClassification: {len(strong)} strong, {len(weak)} weak, {len(none_)} no-cue")


def compute_optimal_weights(transition_data: list[dict], signals: dict, fps: float):
    """Find optimal weights via brute-force grid search.

    Objective: maximize separation between strong-cue and non-cue transitions.
    Also tries adding extra features.
    """
    signal_names = ["p_surge", "p_spread", "p_height", "p_upward", "p_coil"]

    # Build feature matrix: max of each signal in [-4s, +2s] window around each transition
    n_trans = len(transition_data)
    X = np.zeros((n_trans, len(signal_names)))
    for i, entry in enumerate(transition_data):
        for j, sig in enumerate(signal_names):
            X[i, j] = entry[f"{sig}_max"]

    labels = np.array([1 if e["cue_class"] == "strong" else 0 for e in transition_data])

    print("\n" + "=" * 80)
    print("WEIGHT OPTIMIZATION")
    print("=" * 80)

    # Current weights
    current_w = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    current_scores = X @ current_w
    print(f"\nCurrent weights: {dict(zip(signal_names, current_w))}")
    print(f"  Strong cue scores: {current_scores[labels==1]}")
    print(f"  Non-cue scores:    {current_scores[labels==0]}")
    if labels.sum() > 0 and (1-labels).sum() > 0:
        sep = current_scores[labels==1].min() - current_scores[labels==0].max()
        print(f"  Separation (min_strong - max_noncue): {sep:.3f}")

    # Grid search over weight space (weights must sum to 1, all >= 0)
    best_sep = -999
    best_w = current_w.copy()
    best_auc = 0

    step = 0.05
    n_signals = len(signal_names)

    # Generate all weight combinations summing to 1.0 with step size
    from itertools import product
    levels = np.arange(0, 1.0 + step/2, step)

    best_results = []
    for w0 in levels:
        for w1 in levels:
            for w2 in levels:
                for w3 in levels:
                    w4 = 1.0 - w0 - w1 - w2 - w3
                    if w4 < -0.001 or w4 > 1.001:
                        continue
                    w4 = max(0, w4)
                    weights = np.array([w0, w1, w2, w3, w4])
                    scores = X @ weights

                    if labels.sum() == 0 or (1-labels).sum() == 0:
                        continue

                    # Objective: maximize mean separation
                    mean_strong = scores[labels == 1].mean()
                    mean_noncue = scores[labels == 0].mean()
                    min_strong = scores[labels == 1].min()
                    max_noncue = scores[labels == 0].max()

                    # Penalize overlap
                    separation = min_strong - max_noncue
                    mean_sep = mean_strong - mean_noncue

                    # Combined objective
                    obj = 0.5 * separation + 0.5 * mean_sep

                    if obj > best_sep:
                        best_sep = obj
                        best_w = weights.copy()

                    best_results.append((obj, weights.copy(), separation, mean_sep))

    best_results.sort(key=lambda x: -x[0])

    print(f"\nTop 5 weight configurations (by separation):")
    for rank, (obj, w, sep, msep) in enumerate(best_results[:5]):
        wdict = {name: f"{v:.2f}" for name, v in zip(signal_names, w)}
        scores = X @ w
        print(f"  #{rank+1}: {wdict}  sep={sep:.3f}  mean_sep={msep:.3f}")
        if labels.sum() > 0:
            print(f"       Strong scores: {scores[labels==1].round(3)}")
        if (1-labels).sum() > 0:
            print(f"       Non-cue scores: {scores[labels==0].round(3)}")

    print(f"\nBest weights: {dict(zip(signal_names, best_w.round(3)))}")

    return best_w, signal_names


def plot_transition_profiles(
    signals: dict[str, np.ndarray],
    timeline: list[tuple[str, float, int]],
    transition_data: list[dict],
    fps: float,
    total_frames: int,
):
    """Generate matplotlib plots of signal profiles around each transition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    peak_signals = ["p_surge", "p_spread", "p_height", "p_upward", "p_coil"]
    labels = ["Speed surge", "Spread velocity", "Hands raised", "Upward motion", "Stillness-burst"]
    colors = ["#ff6b6b", "#51cf66", "#339af0", "#cc5de8", "#fcc419"]

    wb = int(4.0 * fps)
    wa = int(2.0 * fps)

    # ── Individual transition plots ─────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes = axes.flatten()

    for idx, (entry, (seg_id, t, frame)) in enumerate(zip(transition_data, timeline)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        start = max(0, frame - wb)
        end = min(total_frames, frame + wa)
        time_axis = (np.arange(start, end) - frame) / fps

        for sig_name, label, color in zip(peak_signals, labels, colors):
            ax.plot(time_axis, signals[sig_name][start:end], label=label, color=color, linewidth=1.5)

        # Combined score
        score = (
            0.25 * signals["p_surge"][start:end]
            + 0.25 * signals["p_spread"][start:end]
            + 0.20 * signals["p_height"][start:end]
            + 0.15 * signals["p_upward"][start:end]
            + 0.15 * signals["p_coil"][start:end]
        )
        ax.plot(time_axis, score, "k-", linewidth=2.5, label="Combined", alpha=0.7)
        ax.axhline(0.38, color="red", linestyle="--", alpha=0.4, label="Threshold")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        cls = entry["cue_class"]
        ax.set_title(f"Seg {seg_id} ({int(t//60)}:{t%60:04.1f}s) [{cls}]",
                     fontweight="bold" if cls == "strong" else "normal",
                     color="red" if cls == "strong" else ("orange" if cls == "weak" else "gray"))
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Time relative to transition (s)")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for idx in range(len(timeline), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Signal Profiles Around Each Firebird Transition", fontsize=16)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "transition_profiles.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'transition_profiles.png'}")
    plt.close(fig)

    # ── Average profiles: strong vs weak vs none ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls_idx, cls in enumerate(["strong", "weak", "none"]):
        ax = axes[cls_idx]
        class_entries = [(e, tl) for e, tl in zip(transition_data, timeline) if e["cue_class"] == cls]
        if not class_entries:
            ax.set_title(f"{cls.upper()} cues (n=0)")
            continue

        # Average each signal
        window_len = wb + wa
        for sig_name, label, color in zip(peak_signals, labels, colors):
            all_windows = []
            for entry, (seg_id, t, frame) in class_entries:
                start = max(0, frame - wb)
                end = min(total_frames, frame + wa)
                window = signals[sig_name][start:end]
                # Pad/trim to uniform length
                if len(window) < window_len:
                    window = np.pad(window, (0, window_len - len(window)), mode="edge")
                else:
                    window = window[:window_len]
                all_windows.append(window)

            mean_window = np.mean(all_windows, axis=0)
            std_window = np.std(all_windows, axis=0) if len(all_windows) > 1 else np.zeros_like(mean_window)
            time_axis = (np.arange(window_len) - wb) / fps

            ax.plot(time_axis, mean_window, label=label, color=color, linewidth=2)
            ax.fill_between(time_axis, mean_window - std_window, mean_window + std_window,
                           color=color, alpha=0.15)

        ax.axhline(0.38, color="red", linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{cls.upper()} cues (n={len(class_entries)})", fontsize=14)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Time relative to transition (s)")
        if cls_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Average Signal Profiles by Cue Class", fontsize=16)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "average_profiles.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'average_profiles.png'}")
    plt.close(fig)

    # ── Signal correlation heatmap ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    n_trans = len(transition_data)
    X = np.zeros((n_trans, len(peak_signals)))
    for i, entry in enumerate(transition_data):
        for j, sig in enumerate(peak_signals):
            X[i, j] = entry[f"{sig}_max"]

    corr = np.corrcoef(X.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax)
    ax.set_title("Signal Correlation (max values around transitions)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'correlation_heatmap.png'}")
    plt.close(fig)

    # ── Full timeline overview ──────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(24, 14), sharex=True)
    time_axis = np.arange(total_frames) / fps

    for i, (sig_name, label, color) in enumerate(zip(peak_signals, labels, colors)):
        axes[i].plot(time_axis, signals[sig_name], color=color, linewidth=0.5)
        axes[i].set_ylabel(label, fontsize=9)
        for entry in transition_data:
            c = {"strong": "red", "weak": "orange", "none": "gray"}[entry["cue_class"]]
            axes[i].axvline(entry["time"], color=c, alpha=0.6, linewidth=1)

    # Combined score
    score = (
        0.25 * signals["p_surge"]
        + 0.25 * signals["p_spread"]
        + 0.20 * signals["p_height"]
        + 0.15 * signals["p_upward"]
        + 0.15 * signals["p_coil"]
    )
    axes[5].plot(time_axis, score, "k-", linewidth=0.5)
    axes[5].axhline(0.38, color="red", linestyle="--", alpha=0.4)
    axes[5].set_ylabel("Combined Score")
    for entry in transition_data:
        c = {"strong": "red", "weak": "orange", "none": "gray"}[entry["cue_class"]]
        axes[5].axvline(entry["time"], color=c, alpha=0.6, linewidth=1)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Full Timeline: Signals + Transitions (red=strong, orange=weak, gray=none)", fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "full_timeline.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'full_timeline.png'}")
    plt.close(fig)

    # ── Bar chart: signal max values per transition ─────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_trans)
    width = 0.15
    for j, (sig, label, color) in enumerate(zip(peak_signals, labels, colors)):
        vals = [entry[f"{sig}_max"] for entry in transition_data]
        ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([e["seg_id"] for e in transition_data])
    ax.set_xlabel("Segment")
    ax.set_ylabel("Peak signal value in [-4s, +2s] window")
    ax.legend()
    ax.set_title("Peak Signal Values Per Transition")

    # Mark strong cues
    for i, entry in enumerate(transition_data):
        if entry["cue_class"] == "strong":
            ax.annotate("STRONG", (i + 2 * width, 1.02), ha="center", fontsize=8,
                        color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "signal_bars.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'signal_bars.png'}")
    plt.close(fig)


def extra_feature_analysis(
    kps: np.ndarray,
    signals: dict[str, np.ndarray],
    timeline: list[tuple[str, float, int]],
    transition_data: list[dict],
    fps: float,
    total_frames: int,
):
    """Explore additional features beyond the current 5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = total_frames

    l_wrist = kps[:, L_WRIST, :2]
    r_wrist = kps[:, R_WRIST, :2]
    l_elbow = kps[:, L_ELBOW, :2]
    r_elbow = kps[:, R_ELBOW, :2]
    l_shoulder = kps[:, L_SHOULDER, :2]
    r_shoulder = kps[:, R_SHOULDER, :2]
    nose = kps[:, NOSE, :2]

    # ── Extra feature 1: Elbow angle (arm extension) ─────────────────
    def angle_at_elbow(shoulder, elbow, wrist):
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_a = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        )
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    l_elbow_angle = angle_at_elbow(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = angle_at_elbow(r_shoulder, r_elbow, r_wrist)
    max_elbow_angle = np.maximum(l_elbow_angle, r_elbow_angle)
    # Rate of change of elbow angle (arm extending rapidly)
    elbow_angle_vel = np.abs(np.diff(max_elbow_angle, prepend=max_elbow_angle[0]))

    # ── Extra feature 2: Vertical acceleration ───────────────────────
    l_dy = np.diff(l_wrist[:, 1], prepend=l_wrist[0, 1])
    r_dy = np.diff(r_wrist[:, 1], prepend=r_wrist[0, 1])
    combined_vy = l_dy + r_dy
    vy_accel = np.abs(np.diff(combined_vy, prepend=combined_vy[0]))

    # ── Extra feature 3: Asymmetry between hands ────────────────────
    l_speed = np.linalg.norm(np.diff(l_wrist, axis=0, prepend=l_wrist[:1]), axis=1)
    r_speed = np.linalg.norm(np.diff(r_wrist, axis=0, prepend=r_wrist[:1]), axis=1)
    speed_asymmetry = np.abs(l_speed - r_speed) / (l_speed + r_speed + 1e-8)

    # ── Extra feature 4: Hand distance from body center ─────────────
    body_center = (l_shoulder + r_shoulder) / 2
    l_dist = np.linalg.norm(l_wrist - body_center, axis=1)
    r_dist = np.linalg.norm(r_wrist - body_center, axis=1)
    max_hand_reach = np.maximum(l_dist, r_dist)
    shoulder_w = np.linalg.norm(l_shoulder - r_shoulder, axis=1)
    normalized_reach = max_hand_reach / (shoulder_w + 1e-8)

    # ── Extra feature 5: Jerk (derivative of acceleration) ──────────
    combined_speed = signals["combined_speed"]
    accel = np.diff(combined_speed, prepend=combined_speed[0])
    jerk = np.abs(np.diff(accel, prepend=accel[0]))

    # ── Extra feature 6: Wrist trajectory curvature ─────────────────
    # High curvature = sharp direction change
    l_vel = np.diff(l_wrist, axis=0, prepend=l_wrist[:1])
    l_vel_norm = l_vel / (np.linalg.norm(l_vel, axis=1, keepdims=True) + 1e-8)
    l_curvature = np.linalg.norm(np.diff(l_vel_norm, axis=0, prepend=l_vel_norm[:1]), axis=1)
    r_vel = np.diff(r_wrist, axis=0, prepend=r_wrist[:1])
    r_vel_norm = r_vel / (np.linalg.norm(r_vel, axis=1, keepdims=True) + 1e-8)
    r_curvature = np.linalg.norm(np.diff(r_vel_norm, axis=0, prepend=r_vel_norm[:1]), axis=1)
    max_curvature = np.maximum(l_curvature, r_curvature)

    extra_features = {
        "elbow_angle_vel": elbow_angle_vel,
        "vy_accel": vy_accel,
        "speed_asymmetry": speed_asymmetry,
        "normalized_reach": normalized_reach,
        "jerk": jerk,
        "curvature": max_curvature,
    }
    extra_labels = {
        "elbow_angle_vel": "Elbow extension rate",
        "vy_accel": "Vertical acceleration",
        "speed_asymmetry": "Hand speed asymmetry",
        "normalized_reach": "Hand reach (normalized)",
        "jerk": "Jerk (speed change rate)",
        "curvature": "Trajectory curvature",
    }

    # Smooth extra features with EMA
    def ema_smooth(x, alpha=0.1):
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = out[i-1] + alpha * (x[i] - out[i-1])
        return out

    wb = int(4.0 * fps)
    wa = int(2.0 * fps)

    # Extract extra feature windows around transitions and compute discriminative power
    print("\n" + "=" * 80)
    print("EXTRA FEATURE ANALYSIS")
    print("=" * 80)

    labels_arr = np.array([1 if e["cue_class"] == "strong" else 0 for e in transition_data])

    print(f"\n{'Feature':<25} {'Strong mean':>12} {'Non-cue mean':>13} {'Ratio':>8} {'Rank':>6}")
    print("-" * 70)

    feature_scores = []
    for feat_name, feat_vals in extra_features.items():
        smoothed = ema_smooth(feat_vals, alpha=0.1)
        # Max in window around each transition
        maxes = []
        for entry in transition_data:
            start = max(0, entry["frame"] - wb)
            end = min(n, entry["frame"] + wa)
            maxes.append(np.max(smoothed[start:end]))
        maxes = np.array(maxes)

        strong_mean = maxes[labels_arr == 1].mean() if labels_arr.sum() > 0 else 0
        noncue_mean = maxes[labels_arr == 0].mean() if (1 - labels_arr).sum() > 0 else 0
        ratio = strong_mean / (noncue_mean + 1e-8)
        feature_scores.append((feat_name, strong_mean, noncue_mean, ratio))

    feature_scores.sort(key=lambda x: -x[3])
    for rank, (name, sm, nm, ratio) in enumerate(feature_scores, 1):
        print(f"{extra_labels[name]:<25} {sm:>12.2f} {nm:>13.2f} {ratio:>8.2f} {rank:>6}")

    # Plot extra features around transitions
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (feat_name, feat_vals) in enumerate(extra_features.items()):
        ax = axes[idx]
        smoothed = ema_smooth(feat_vals, alpha=0.1)

        for entry, (seg_id, t, frame) in zip(transition_data, timeline):
            start = max(0, frame - wb)
            end = min(n, frame + wa)
            time_axis = (np.arange(start, end) - frame) / fps
            c = {"strong": "red", "weak": "orange", "none": "gray"}[entry["cue_class"]]
            ax.plot(time_axis, smoothed[start:end], color=c, alpha=0.6, linewidth=1.5)

        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(extra_labels[feat_name])
        ax.set_xlabel("Time relative to transition (s)")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "extra_features.png", dpi=150)
    print(f"\nSaved: {RESULTS_DIR / 'extra_features.png'}")
    plt.close(fig)

    return extra_features


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--process-every", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Reuse cached keypoints (.keypoints_cache.npz)")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if not CACHE_FILE.exists() and args.video is None:
        parser.error("--video is required when no keypoint cache exists; alternatively set FIREBIRD_VIDEO")

    if args.clear_cache and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("Cache cleared.")

    # 1. Extract keypoints
    kps, fps, total_frames = extract_keypoints(args.video, args.process_every, args.imgsz)
    print(f"Keypoints shape: {kps.shape}, fps={fps}, frames={total_frames}")

    # 2. Parse timeline
    timeline = parse_timeline(TIMELINE_FILE, fps)
    print(f"\nTimeline: {len(timeline)} transitions")
    for seg_id, t, frame in timeline:
        print(f"  Seg {seg_id} at {t:.2f}s (frame {frame})")

    # 3. Compute signals
    print("\nComputing signals...")
    signals = compute_signals(kps, fps)

    # 4. Analyze transitions
    print("\nAnalyzing transitions...")
    transition_data = analyze_transitions(signals, timeline, fps, total_frames)
    transition_data = classify_cues(transition_data)
    print_summary(transition_data)

    # 5. Optimize weights
    best_w, sig_names = compute_optimal_weights(transition_data, signals, fps)

    # 6. Extra feature analysis
    extra_features = extra_feature_analysis(kps, signals, timeline, transition_data, fps, total_frames)

    # 7. Generate plots
    print("\nGenerating plots...")
    plot_transition_profiles(signals, timeline, transition_data, fps, total_frames)

    # 8. Save results JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    results_json = {
        "transitions": transition_data,
        "current_weights": {"p_surge": 0.25, "p_spread": 0.25, "p_height": 0.20,
                           "p_upward": 0.15, "p_coil": 0.15},
        "optimized_weights": dict(zip(sig_names, [round(float(w), 3) for w in best_w])),
    }
    with open(RESULTS_DIR / "analysis_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'analysis_results.json'}")

    print("\nDone! Check analysis_results/ for plots and data.")


if __name__ == "__main__":
    main()
