#!/usr/bin/env python3
"""Richer PCA on Hardy's 22 cues using the live detector's causal windows.

Goals:
  - Use the SAME windows as the live RF detector: deep_pre [-2.5, -1.0],
    prep [-1.0, 0], now [-0.4, 0]. All causal, no look-ahead.
  - Throw a much larger kinematic feature catalogue at PCA and see if the
    component structure changes.
  - Compare with the original 6-stat-per-joint design to check whether we
    are missing important factors.

Per joint (L/R wrist, L/R elbow), per window:
  16 features covering position, velocity, acceleration, jerk, direction,
  path geometry, displacement, end-state.

Plus cross-joint features per window:
  hands-spread stats, L-R coordination, elbow-wrist distance.

Total target: ~216 features for 22 cues. Yes, p > n. PCA still works, we
just look at the variance and loadings, not run a model on top.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

CACHE = Path(__file__).parent / ".keypoints_cache_xl.npz"
ANNOT = Path(__file__).parent / "uncle_annotations.json"
OUT = Path(__file__).parent / "analysis_results"
OUT.mkdir(exist_ok=True)

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

JOINTS = [
    ("L_wrist", L_WRIST),
    ("R_wrist", R_WRIST),
    ("L_elbow", L_ELBOW),
    ("R_elbow", R_ELBOW),
]

# SAME windows as live_detector.py
WINDOWS = [
    ("deep_pre", -2.5, -1.0),
    ("prep", -1.0, 0.0),
    ("now", -0.4, 0.0),
]


def body_norm(kps_win):
    l_sh = kps_win[:, L_SHOULDER, :2]
    r_sh = kps_win[:, R_SHOULDER, :2]
    centre = (l_sh + r_sh) / 2.0
    sh_w = np.maximum(np.linalg.norm(l_sh - r_sh, axis=1), 1.0)
    return {
        name: (kps_win[:, jid, :2] - centre) / sh_w[:, None]
        for name, jid in JOINTS
    }


JOINT_STAT_NAMES = [
    # Position stats (8)
    "mean_x", "mean_y", "std_x", "std_y",
    "range_x", "range_y", "iqr_x", "iqr_y",
    # Velocity magnitude stats (8)
    "speed_mean", "speed_max", "speed_min", "speed_std",
    "speed_median", "speed_iqr",
    "speed_first_half_mean", "speed_second_half_mean",
    # Velocity component stats (8)
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
assert len(JOINT_STAT_NAMES) == 64, len(JOINT_STAT_NAMES)


def joint_features_rich(traj):
    """64 statistics per joint trajectory (Nx2 array). Order matches
    JOINT_STAT_NAMES exactly."""
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

    # Path stats
    path_len = float(speed.sum())
    disp_vec = traj[-1] - traj[0]
    disp_x = float(disp_vec[0])
    disp_y = float(disp_vec[1])
    disp_mag = float(np.linalg.norm(disp_vec))
    straight = disp_mag / max(path_len, 1e-6)

    # Direction asymmetric totals (sum of one-sided components)
    ascent_total = float(np.maximum(0.0, -dy).sum())
    descent_total = float(np.maximum(0.0, dy).sum())
    left_total = float(np.maximum(0.0, -dx).sum())
    right_total = float(np.maximum(0.0, dx).sum())
    net_progress = float(np.linalg.norm(np.array([dx.sum(), dy.sum()])))

    # Time-localized
    if len(speed):
        argmax_speed = float(int(np.argmax(speed)) / max(len(speed) - 1, 1))
        argmin_speed = float(int(np.argmin(speed)) / max(len(speed) - 1, 1))
    else:
        argmax_speed = argmin_speed = 0.0
    if len(speed) > 1:
        # Linear trend of speed
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
    # Peak localization: how far the centroid is from the mean position
    centroid = traj.mean(axis=0)
    peak_loc = float(np.max(np.linalg.norm(traj - centroid, axis=1)))
    centroid_dist = float(np.linalg.norm(centroid))

    # accel/jerk at peak speed
    if len(speed) and len(acc_mag):
        i_peak = int(np.argmax(speed))
        ai = min(i_peak, len(acc_mag) - 1)
        accel_at_peak = float(acc_mag[ai])
    else:
        accel_at_peak = 0.0
    if len(speed) and len(jrk_mag):
        i_peak = int(np.argmax(speed))
        ji = min(i_peak, len(jrk_mag) - 1)
        jerk_at_peak = float(jrk_mag[ji])
    else:
        jerk_at_peak = 0.0

    feats = [
        # Position (8)
        float(x.mean()), float(y.mean()),
        float(x.std()), float(y.std()),
        float(x.max() - x.min()), float(y.max() - y.min()),
        float(np.percentile(x, 75) - np.percentile(x, 25)),
        float(np.percentile(y, 75) - np.percentile(y, 25)),
        # Velocity magnitude (8)
        float(speed.mean()), float(speed.max()), float(speed.min()),
        float(speed.std()), float(np.median(speed)),
        float(np.percentile(speed, 75) - np.percentile(speed, 25)),
        float(speed[:half].mean()), float(speed[half:].mean()),
        # Velocity component (8)
        float(dx.mean()), float(dy.mean()),
        float(dx.std()), float(dy.std()),
        float(dx.max() - dx.min()), float(dy.max() - dy.min()),
        float(np.abs(dx).mean()), float(np.abs(dy).mean()),
        # Direction peaks (8)
        float(np.maximum(0.0, -dy).max()),  # up_peak
        float(np.maximum(0.0, dy).max()),   # down_peak
        float(np.maximum(0.0, -dx).max()),  # left_peak
        float(np.maximum(0.0, dx).max()),   # right_peak
        float(np.maximum(0.0, -dy).mean()),
        float(np.maximum(0.0, dy).mean()),
        float(np.maximum(0.0, -dx).mean()),
        float(np.maximum(0.0, dx).mean()),
        # Acceleration (6)
        float(acc_mag.mean()), float(acc_mag.max()), float(acc_mag.std()),
        float(acc[:, 1].mean()) if len(acc) else 0.0,
        float(acc[:, 0].mean()) if len(acc) else 0.0,
        accel_at_peak,
        # Jerk (4)
        float(jrk_mag.mean()), float(jrk_mag.max()), float(jrk_mag.std()),
        jerk_at_peak,
        # Path geometry (8)
        path_len, straight, disp_x, disp_y, disp_mag,
        ascent_total, descent_total, net_progress,
        # Time-localized (8)
        argmax_speed, argmin_speed,
        speed_trend, accel_trend, y_trend, x_trend,
        peak_loc, centroid_dist,
        # End-state (6)
        float(speed[-1]) if len(speed) else 0.0,
        float(dx[-1]) if len(dx) else 0.0,
        float(dy[-1]) if len(dy) else 0.0,
        float(acc_mag[-1]) if len(acc_mag) else 0.0,
        float(x[-1]), float(y[-1]),
    ]
    assert len(feats) == 64, len(feats)
    return feats


def cross_joint_features(norm):
    """8 features per window for cross-joint coordination."""
    Lw = norm["L_wrist"]
    Rw = norm["R_wrist"]
    Le = norm["L_elbow"]
    Re = norm["R_elbow"]
    if len(Lw) < 4:
        return [0.0] * 8

    spread = np.linalg.norm(Lw - Rw, axis=1)
    spread_dvel = np.diff(spread) if len(spread) > 1 else np.array([0.0])

    # L-R wrist y correlation
    yL = Lw[:, 1] - Lw[:, 1].mean()
    yR = Rw[:, 1] - Rw[:, 1].mean()
    denom = np.sqrt((yL ** 2).sum() * (yR ** 2).sum()) + 1e-9
    corr_y = float((yL * yR).sum() / denom)

    # mean elbow-wrist distance per side
    Ldist = np.linalg.norm(Lw - Le, axis=1).mean()
    Rdist = np.linalg.norm(Rw - Re, axis=1).mean()

    return [
        float(spread.mean()),               # 1 spread_mean
        float(spread.std()),                # 2 spread_std
        float(spread.max()),                # 3 spread_max
        float(np.abs(spread_dvel).max()),   # 4 spread_dvel_max
        float(spread[-1]),                  # 5 spread_now
        corr_y,                              # 6 LR_y_corr
        float(Ldist),                        # 7 L_elbow_wrist_dist
        float(Rdist),                        # 8 R_elbow_wrist_dist
    ]


CROSS_NAMES = [
    "spread_mean", "spread_std", "spread_max", "spread_dvel_max",
    "spread_now", "LR_y_corr", "L_elbow_wrist_dist", "R_elbow_wrist_dist",
]


def cue_feature_vector(kps, cue_time, fps, total):
    feats: list[float] = []
    feat_names: list[str] = []
    for wn, ws, we in WINDOWS:
        f0 = int((cue_time + ws) * fps)
        f1 = int((cue_time + we) * fps)
        s = max(0, f0)
        e = min(total, f1)
        for jn, _ in JOINTS:
            for stat in JOINT_STAT_NAMES:
                feat_names.append(f"{wn}/{jn}/{stat}")
        for nm in CROSS_NAMES:
            feat_names.append(f"{wn}/{nm}")
        if e - s < 4:
            feats += [0.0] * (len(JOINTS) * len(JOINT_STAT_NAMES) + len(CROSS_NAMES))
            continue
        norm = body_norm(kps[s:e])
        for jn, _ in JOINTS:
            feats += joint_features_rich(norm[jn])
        feats += cross_joint_features(norm)
    return np.asarray(feats), feat_names


def run_pca(X):
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    cov = np.cov(Xs.T)
    ev, evc = np.linalg.eigh(cov)
    order = np.argsort(-ev)
    ev = ev[order]
    evc = evc[:, order]
    explained = ev / ev.sum()
    return Xs @ evc, explained, evc


def breakdown(loadings_abs2, names, key):
    """Sum |loading|^2 over names whose path contains key (e.g. 'L_wrist')."""
    idx = [i for i, n in enumerate(names) if key in n]
    return float(loadings_abs2[idx].sum()), len(idx)


def main():
    print("Loading data…")
    data = np.load(CACHE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])
    cues = json.loads(ANNOT.read_text())["cues"]
    n_cues = len(cues)
    print(f"  kps {kps.shape} @ {fps}fps, {n_cues} cues")

    print("Building rich feature matrix…")
    feat_mats = []
    feat_names = None
    for c in cues:
        v, names = cue_feature_vector(kps, c["time"], fps, total)
        feat_mats.append(v)
        if feat_names is None:
            feat_names = names
    X = np.array(feat_mats)
    print(f"  X shape: {X.shape}  ({X.shape[1]} features for {n_cues} cues — p > n)")

    X_pca, explained, evc = run_pca(X)
    print(f"\nPCA variance explained:")
    cum = 0.0
    for i in range(min(10, len(explained))):
        cum = explained[: i + 1].sum()
        print(f"  PC{i+1}: {explained[i]:>6.1%}  (cum {cum:>6.1%})")

    n_keep_90 = int(np.argmax(explained.cumsum() >= 0.90) + 1)
    print(f"\n  Need {n_keep_90} PCs to reach 90% variance")

    # Loadings: for each PC, break down by joint / stat / window
    print("\n" + "=" * 80)
    for pc_idx in range(min(5, evc.shape[1])):
        L = np.abs(evc[:, pc_idx])
        L2 = L ** 2
        print(f"\n=== PC{pc_idx+1} ({explained[pc_idx]:.1%}) ===")

        print("  By WINDOW (Σ |loading|² over all joints × stats):")
        for wn, _, _ in WINDOWS:
            s, k = breakdown(L2, feat_names, f"{wn}/")
            print(f"    {wn:<10}  {s:.3f}   ({k} features)")

        print("  By JOINT (Σ |loading|² over all windows × stats, joint features only):")
        for jn, _ in JOINTS:
            s, k = breakdown(L2, feat_names, f"/{jn}/")
            print(f"    {jn:<10}  {s:.3f}   ({k} features)")

        print("  By STAT (joint features):")
        for stat in JOINT_STAT_NAMES:
            s, k = breakdown(L2, feat_names, f"/{stat}")
            print(f"    {stat:<14} {s:.3f}   ({k} features)")

        print("  Cross-joint features mass:")
        for nm in CROSS_NAMES:
            s, _ = breakdown(L2, feat_names, f"/{nm}")
            print(f"    {nm:<22} {s:.3f}")

        print("  Top-10 individual loadings:")
        top = np.argsort(-L)[:10]
        for ix in top:
            print(f"    {feat_names[ix]:<46} {evc[ix, pc_idx]:+.3f}")

    # Quick clustering for comparison
    print("\n" + "=" * 80)
    print("Hierarchical clustering on first PCs covering 90%")
    Z = linkage(pdist(X_pca[:, :n_keep_90]), method="ward")
    for K in [3, 4, 5]:
        cl = fcluster(Z, t=K, criterion="maxclust")
        sizes = list(np.bincount(cl)[1:])
        print(f"  k={K}: sizes {sizes}")

    # Compare with original 6-stat PCA
    print("\n" + "=" * 80)
    print("Comparison with original 6-stat-per-joint, 4-window PCA")
    print("  (importing combined_features for an apples-to-apples re-run)")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from combined_features import (
        WINDOWS as ORIG_WINDOWS,
        window_features as orig_features,
    )
    orig_feats = []
    orig_names = None
    for c in cues:
        row = {}
        for wn, ws, we in ORIG_WINDOWS:
            f = orig_features(kps, c["time"], fps, total, ws, we) or {}
            for k, v in f.items():
                row[f"{wn}__{k}"] = v
        if orig_names is None:
            orig_names = list(row.keys())
        orig_feats.append([row.get(k, 0.0) for k in orig_names])
    Xo = np.array(orig_feats)
    _, exp_o, _ = run_pca(Xo)
    print(f"  Original 6-stat × 4 windows: {Xo.shape[1]} features")
    for i in range(5):
        print(f"    PC{i+1}: {exp_o[i]:.1%}")
    print(f"  Rich (16-stat × 3 causal windows + cross): {X.shape[1]} features")
    for i in range(5):
        print(f"    PC{i+1}: {explained[i]:.1%}")

    # ── Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    n_show = min(15, len(explained))
    ax.bar(range(1, n_show + 1), explained[:n_show] * 100,
           color="#3498db", alpha=0.85)
    ax.plot(range(1, n_show + 1), explained[:n_show].cumsum() * 100,
            "ro-", markersize=5)
    ax.axhline(90, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance %")
    ax.set_title(f"Rich PCA scree ({X.shape[1]} feats)", fontweight="bold")
    for i in range(5):
        ax.text(i + 1, explained[i] * 100 + 2, f"{explained[i]:.0%}",
                ha="center", fontsize=8)

    # Compare scree side-by-side
    ax = fig.add_subplot(gs[0, 1])
    n_cmp = min(8, len(exp_o), len(explained))
    width = 0.4
    xpos = np.arange(1, n_cmp + 1)
    ax.bar(xpos - width / 2, exp_o[:n_cmp] * 100, width,
           color="#e67e22", label=f"Original 6-stat (96)")
    ax.bar(xpos + width / 2, explained[:n_cmp] * 100, width,
           color="#3498db", label=f"Rich ({X.shape[1]})")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance %")
    ax.set_title("Variance explained: original vs rich",
                 fontweight="bold")
    ax.legend(fontsize=9)

    # PC scatter
    for col, (xpc, ypc) in enumerate([(0, 1), (1, 2)]):
        ax = fig.add_subplot(gs[0, 2] if col == 0 else gs[1, 0])
        ax.scatter(X_pca[:, xpc], X_pca[:, ypc], s=130, c="#34495e",
                   edgecolors="white", zorder=3)
        for i, c in enumerate(cues):
            mk = "*" if "Start" in c["label"] else str(i)
            ax.annotate(mk, (X_pca[i, xpc], X_pca[i, ypc]),
                        fontsize=8, ha="center", va="center", color="white")
        ax.set_xlabel(f"PC{xpc+1} ({explained[xpc]:.0%})")
        ax.set_ylabel(f"PC{ypc+1} ({explained[ypc]:.0%})")
        ax.set_title(f"PC{xpc+1} vs PC{ypc+1}", fontweight="bold")
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)

    # Window mass per PC (PCs 1..5)
    ax = fig.add_subplot(gs[1, 1])
    pcs_show = min(5, evc.shape[1])
    win_labels = [w[0] for w in WINDOWS]
    mat = np.zeros((pcs_show, len(WINDOWS)))
    for p in range(pcs_show):
        L2 = evc[:, p] ** 2
        for j, wn in enumerate(win_labels):
            s, _ = breakdown(L2, feat_names, f"{wn}/")
            mat[p, j] = s
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels(win_labels)
    ax.set_yticks(range(pcs_show))
    ax.set_yticklabels([f"PC{i+1}" for i in range(pcs_show)])
    ax.set_title("Window mass per PC (Σ |loading|²)",
                 fontweight="bold")
    for p in range(pcs_show):
        for j in range(len(WINDOWS)):
            ax.text(j, p, f"{mat[p, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if mat[p, j] > mat.max() * 0.5 else "black")

    # Stat mass per PC
    ax = fig.add_subplot(gs[1, 2])
    statmat = np.zeros((pcs_show, len(JOINT_STAT_NAMES)))
    for p in range(pcs_show):
        L2 = evc[:, p] ** 2
        for j, st in enumerate(JOINT_STAT_NAMES):
            s, _ = breakdown(L2, feat_names, f"/{st}")
            statmat[p, j] = s
    im = ax.imshow(statmat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(JOINT_STAT_NAMES)))
    ax.set_xticklabels(JOINT_STAT_NAMES, rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(pcs_show))
    ax.set_yticklabels([f"PC{i+1}" for i in range(pcs_show)])
    ax.set_title("Stat mass per PC (joint features)",
                 fontweight="bold")

    # Joint mass per PC
    ax = fig.add_subplot(gs[2, 0])
    jmat = np.zeros((pcs_show, len(JOINTS)))
    for p in range(pcs_show):
        L2 = evc[:, p] ** 2
        for j, (jn, _) in enumerate(JOINTS):
            s, _ = breakdown(L2, feat_names, f"/{jn}/")
            jmat[p, j] = s
    ax.imshow(jmat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(JOINTS)))
    ax.set_xticklabels([j[0] for j in JOINTS])
    ax.set_yticks(range(pcs_show))
    ax.set_yticklabels([f"PC{i+1}" for i in range(pcs_show)])
    ax.set_title("Joint mass per PC", fontweight="bold")
    for p in range(pcs_show):
        for j in range(len(JOINTS)):
            ax.text(j, p, f"{jmat[p, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if jmat[p, j] > jmat.max() * 0.5
                    else "black")

    # Cross-joint mass per PC
    ax = fig.add_subplot(gs[2, 1])
    cmat = np.zeros((pcs_show, len(CROSS_NAMES)))
    for p in range(pcs_show):
        L2 = evc[:, p] ** 2
        for j, nm in enumerate(CROSS_NAMES):
            s, _ = breakdown(L2, feat_names, f"/{nm}")
            cmat[p, j] = s
    ax.imshow(cmat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(CROSS_NAMES)))
    ax.set_xticklabels(CROSS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pcs_show))
    ax.set_yticklabels([f"PC{i+1}" for i in range(pcs_show)])
    ax.set_title("Cross-joint feature mass per PC", fontweight="bold")

    # Note about p > n
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    note = (
        f"X shape: {X.shape[0]} cues × {X.shape[1]} features\n\n"
        f"With p > n the covariance matrix is rank ≤ n-1, so we get at most "
        f"{n_cues-1} non-trivial PCs. PCA still produces meaningful directions; "
        f"the variance is just spread across more components.\n\n"
        f"Original 6-stat-per-joint × 4 windows used 96 features and got "
        f"PC1={exp_o[0]:.0%}, PC2={exp_o[1]:.0%}, PC3={exp_o[2]:.0%}.\n\n"
        f"Rich {X.shape[1]} features × 3 causal windows: "
        f"PC1={explained[0]:.0%}, PC2={explained[1]:.0%}, PC3={explained[2]:.0%}.\n\n"
        f"If the structure changed → original feature set was missing a factor.\n"
        f"If similar → original 6 stats already captured the dominant signal."
    )
    ax.text(0, 1, note, fontsize=9, va="top", color="#222",
            wrap=True)

    fig.suptitle(
        "Rich PCA on causal windows — does adding features change the picture?",
        fontsize=14, fontweight="bold")
    fig.savefig(OUT / "figure_pca_rich.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved figure: {OUT / 'figure_pca_rich.png'}")


if __name__ == "__main__":
    main()
