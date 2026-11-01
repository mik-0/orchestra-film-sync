#!/usr/bin/env python3
"""Combined multi-window features + PCA + clustering on clean yolov8s data.

For each cue, extract features from 3 different time windows, each
capturing a different aspect of the gesture:

  1. DEEP PRE ([-2.5s, -1.0s]):  "what was the conductor doing before
     the preparation started?"  Captures stillness / sustained gestures.
  2. PREP    ([-1.0s, 0s]):     "how did the preparation build up?"
     Captures the run-up energy and direction.
  3. CUE     ([0s, +1.0s]):     "what did the cue look like?"
     Captures the main gesture itself.
  4. POST    ([+1.0s, +1.5s]):  "how did the gesture resolve?"
     Captures follow-through and stopping behaviour.

Each window yields scalar summary features (stillness, mean speed,
max upward velocity, range of motion, etc.) -- interpretable -- plus
a smooth DCT representation of the window for shape.

PCA on the combined feature vector, then cluster.

Usage:
    python3 combined_features.py
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
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CACHE_FILE = Path(__file__).parent / ".keypoints_cache_xl.npz"
ANNOT_FILE = Path(__file__).parent / "uncle_annotations.json"
RESULTS_DIR = Path(__file__).parent / "analysis_results"
RESULTS_DIR.mkdir(exist_ok=True)

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

WINDOWS = [
    ("deep_pre", -2.5, -1.0),
    ("prep",     -1.0,  0.0),
    ("cue",       0.0,  1.0),
    ("post",      1.0,  1.5),
]


def window_frames(cue_time, fps, start_s, end_s, total):
    f0 = int((cue_time + start_s) * fps)
    f1 = int((cue_time + end_s) * fps)
    s = max(0, f0)
    e = min(total, f1)
    return s, e


def body_norm(kps_window):
    l_sh = kps_window[:, L_SHOULDER, :2]
    r_sh = kps_window[:, R_SHOULDER, :2]
    centre = (l_sh + r_sh) / 2
    sh_w = np.maximum(np.linalg.norm(l_sh - r_sh, axis=1), 1.0)
    joints = {}
    for name, jid in [("L_wrist", L_WRIST), ("R_wrist", R_WRIST),
                       ("L_elbow", L_ELBOW), ("R_elbow", R_ELBOW)]:
        xy = kps_window[:, jid, :2]
        joints[name] = (xy - centre) / sh_w[:, None]
    return joints


def window_features(kps, cue_time, fps, total, w_start, w_end):
    s, e = window_frames(cue_time, fps, w_start, w_end, total)
    if e - s < 3:
        return None
    norm = body_norm(kps[s:e])

    feats = {}
    for name, traj in norm.items():
        # stillness (position std)
        feats[f"{name}_still"] = float(np.std(traj, axis=0).sum())
        # speed mean + max
        vel = np.diff(traj, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        feats[f"{name}_speed_mean"] = float(speed.mean()) if len(speed) else 0.0
        feats[f"{name}_speed_max"] = float(speed.max()) if len(speed) else 0.0
        # upward peak (negative dy = going up)
        dy = vel[:, 1]
        feats[f"{name}_up_peak"] = float(np.maximum(0, -dy).max()) if len(dy) else 0.0
        # vertical range
        feats[f"{name}_y_range"] = float(traj[:, 1].max() - traj[:, 1].min())
        # horizontal range
        feats[f"{name}_x_range"] = float(traj[:, 0].max() - traj[:, 0].min())
    return feats


def main():
    data = np.load(CACHE_FILE)
    kps = data["keypoints"]
    fps = float(data["fps"])
    total = int(data["total_frames"])

    annot = json.loads(ANNOT_FILE.read_text())
    cues = annot["cues"]
    n_cues = len(cues)
    uncle_labels = [c["label"] for c in cues]

    # Build feature matrix: for each cue, for each window, compute scalar summaries
    feat_rows = []
    feat_names = []
    for i, cue in enumerate(cues):
        row = {}
        for win_name, w_start, w_end in WINDOWS:
            feats = window_features(kps, cue["time"], fps, total, w_start, w_end)
            if feats is None:
                for k in [f"L_wrist_still", f"R_wrist_still", f"L_elbow_still", f"R_elbow_still",
                           f"L_wrist_speed_mean", f"R_wrist_speed_mean",
                           f"L_elbow_speed_mean", f"R_elbow_speed_mean",
                           f"L_wrist_speed_max", f"R_wrist_speed_max",
                           f"L_elbow_speed_max", f"R_elbow_speed_max",
                           f"L_wrist_up_peak", f"R_wrist_up_peak",
                           f"L_wrist_y_range", f"R_wrist_y_range",
                           f"L_wrist_x_range", f"R_wrist_x_range"]:
                    row[f"{win_name}__{k}"] = 0.0
            else:
                for k, v in feats.items():
                    row[f"{win_name}__{k}"] = v
        if i == 0:
            feat_names = list(row.keys())
        feat_rows.append([row[k] for k in feat_names])
    X = np.array(feat_rows)
    print(f"Feature matrix: {X.shape}  ({X.shape[1]} features from {len(WINDOWS)} windows)")

    # Standardize
    mu, sigma = X.mean(0), X.std(0); sigma[sigma < 1e-8] = 1.0
    X_std = (X - mu) / sigma

    # PCA
    cov = np.cov(X_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    explained = eigvals / eigvals.sum()
    X_pca = X_std @ eigvecs

    print("\nPCA variance explained:")
    for i in range(10):
        cum = explained[:i+1].sum()
        print(f"  PC{i+1}: {explained[i]:>6.1%}  (cum {cum:>6.1%})")

    # Top loadings per PC
    print("\nTop features per principal component:")
    for pc in range(min(5, len(eigvecs))):
        loadings = eigvecs[:, pc]
        top = np.argsort(-np.abs(loadings))[:5]
        print(f"\n  PC{pc+1} ({explained[pc]:.1%}):")
        for idx in top:
            print(f"    {feat_names[idx]:<40} {loadings[idx]:+.3f}")

    # Cluster on first n PCs that cover 90%
    cum = explained.cumsum()
    n_keep = int(np.argmax(cum >= 0.90) + 1)
    print(f"\nClustering on first {n_keep} PCs (90% variance)")
    Z = linkage(pdist(X_pca[:, :n_keep]), method="ward")
    results = {}
    for k in range(2, 8):
        cl = fcluster(Z, t=k, criterion="maxclust")
        sizes = list(np.bincount(cl)[1:])
        results[k] = cl
        print(f"  k={k}: sizes {sizes}")

    # Pick k=4 or k=5 as reasonable
    for K in [4, 5]:
        clusters = results[K]
        print(f"\n── k={K} members ──")
        for c in range(1, K + 1):
            members = [i for i in range(n_cues) if clusters[i] == c]
            print(f"  Cluster {c} (n={len(members)}):")
            for i in members:
                print(f"    idx {i:>2} {cues[i]['formatted']:>8}  ({uncle_labels[i]})")

    # ── Figure ────────────────────────────────────────────────────
    K = 5
    clusters = results[K]
    palette = plt.cm.Dark2(np.linspace(0, 1, K))

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3,
                  height_ratios=[1, 1, 1.2])

    # Scree
    ax = fig.add_subplot(gs[0, 0])
    n_show = min(15, len(eigvals))
    ax.bar(range(1, n_show + 1), explained[:n_show] * 100,
           color="#3498db", alpha=0.75)
    ax.plot(range(1, n_show + 1), explained[:n_show].cumsum() * 100, "ro-", markersize=4)
    ax.axhline(90, linestyle=":", color="gray", alpha=0.5)
    ax.set_title("Scree plot", fontweight="bold"); ax.set_xlabel("PC"); ax.set_ylabel("Var %")
    for i in range(5):
        ax.text(i+1, explained[i]*100+2, f"{explained[i]:.0%}", ha="center", fontsize=8)

    # PC1 vs PC2
    ax = fig.add_subplot(gs[0, 1])
    for c in range(1, K + 1):
        mask = clusters == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette[c-1]], s=130,
                   edgecolors="white", zorder=3, label=f"C{c} (n={mask.sum()})")
    for i in range(n_cues):
        ax.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]), fontsize=7, ha="center", va="center")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%})"); ax.set_ylabel(f"PC2 ({explained[1]:.0%})")
    ax.set_title(f"PC1 vs PC2 — clusters (k={K})", fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)

    # PC2 vs PC3
    ax = fig.add_subplot(gs[0, 2])
    for c in range(1, K + 1):
        mask = clusters == c
        ax.scatter(X_pca[mask, 1], X_pca[mask, 2], c=[palette[c-1]], s=130,
                   edgecolors="white", zorder=3)
    for i in range(n_cues):
        ax.annotate(str(i), (X_pca[i, 1], X_pca[i, 2]), fontsize=7, ha="center", va="center")
    ax.set_xlabel(f"PC2 ({explained[1]:.0%})"); ax.set_ylabel(f"PC3 ({explained[2]:.0%})")
    ax.set_title("PC2 vs PC3", fontweight="bold")
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)

    # Uncle labels vs clusters
    ax = fig.add_subplot(gs[0, 3])
    unique_u = sorted(set(uncle_labels))
    M = np.zeros((len(unique_u), K), dtype=int)
    for c in range(1, K + 1):
        for lbl in unique_u:
            M[unique_u.index(lbl), c-1] = sum(1 for i in range(n_cues)
                                              if clusters[i] == c and uncle_labels[i] == lbl)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    for i in range(len(unique_u)):
        for j in range(K):
            if M[i, j] > 0:
                color = "white" if M[i, j] > M.max() * 0.5 else "black"
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
    ax.set_yticks(range(len(unique_u)))
    ax.set_yticklabels([l.split(" / ")[0] if " / " in l else l for l in unique_u], fontsize=8)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"C{c}" for c in range(1, K+1)], fontsize=8)
    ax.set_title("Uncle label vs discovered cluster", fontweight="bold")

    # Dendrogram
    ax = fig.add_subplot(gs[1, 0])
    dendrogram(Z, labels=[str(i) for i in range(n_cues)], ax=ax, leaf_font_size=8,
               color_threshold=Z[-K+1, 2])
    ax.set_title(f"Dendrogram (k={K} threshold)", fontweight="bold")
    ax.set_xlabel("Cue index")

    # Cluster profile: mean normalized feature value per cluster, grouped by window
    # Show only "still" and "speed_max" features
    key_features = [
        ("deep_pre", "L_wrist_still"),
        ("deep_pre", "R_wrist_still"),
        ("prep", "L_wrist_speed_max"),
        ("prep", "R_wrist_speed_max"),
        ("prep", "L_wrist_up_peak"),
        ("cue", "L_wrist_speed_max"),
        ("cue", "L_wrist_y_range"),
        ("cue", "L_wrist_x_range"),
        ("post", "L_wrist_speed_mean"),
    ]
    key_names = [f"{w}.{f}" for w, f in key_features]
    key_idx = [feat_names.index(f"{w}__{f}") for w, f in key_features]

    ax = fig.add_subplot(gs[1, 1:])
    x_pos = np.arange(len(key_features))
    width = 0.8 / K
    for c in range(1, K + 1):
        mask = clusters == c
        if mask.sum() == 0: continue
        means = X_std[mask][:, key_idx].mean(axis=0)
        ax.bar(x_pos + (c - (K+1)/2) * width, means, width,
               color=palette[c-1], alpha=0.85, label=f"C{c}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(key_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Z-score (std from mean)")
    ax.set_title("Cluster profiles on key features (z-scored)", fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=9, loc="best")

    # Row 3: characteristic trajectories per cluster (L wrist y, full window)
    ax1 = fig.add_subplot(gs[2, :2])
    ax2 = fig.add_subplot(gs[2, 2:])

    # For visualization, extract a long trajectory [-2.5, +1.5] for each cue
    LONG_BEFORE = 2.5; LONG_AFTER = 1.5
    LONG_N = 64
    traj_long = np.zeros((n_cues, LONG_N, 2))  # (cue, time, L/R wrist y)
    for i, cue in enumerate(cues):
        frame = int(cue["time"] * fps)
        s = max(0, frame - int(LONG_BEFORE * fps))
        e = min(total, frame + int(LONG_AFTER * fps))
        raw = kps[s:e]
        idx = np.linspace(0, len(raw)-1, LONG_N).astype(int)
        samp = raw[idx]
        l_sh = samp[:, L_SHOULDER, :2]; r_sh = samp[:, R_SHOULDER, :2]
        centre = (l_sh + r_sh) / 2
        sh_w = np.maximum(np.linalg.norm(l_sh - r_sh, axis=1), 1.0)
        traj_long[i, :, 0] = (samp[:, L_WRIST, 1] - centre[:, 1]) / sh_w
        traj_long[i, :, 1] = (samp[:, R_WRIST, 1] - centre[:, 1]) / sh_w

    t_axis = np.linspace(-LONG_BEFORE, LONG_AFTER, LONG_N)

    for ax, joint_col, joint_name in [(ax1, 0, "L wrist"), (ax2, 1, "R wrist")]:
        for c in range(1, K + 1):
            mask = clusters == c
            if mask.sum() == 0: continue
            for mi in np.where(mask)[0]:
                ax.plot(t_axis, -traj_long[mi, :, joint_col],
                        color=palette[c-1], linewidth=0.7, alpha=0.35)
            mean_tr = traj_long[mask, :, joint_col].mean(axis=0)
            ax.plot(t_axis, -mean_tr, color=palette[c-1], linewidth=3,
                    label=f"C{c} mean (n={mask.sum()})")
        ax.axvline(0, color="black", linestyle=":", alpha=0.5)
        ax.axvline(-1, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(1, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"{joint_name} — cluster trajectories over [-2.5s, +1.5s]",
                     fontweight="bold")
        ax.set_xlabel("Time relative to cue label (s)")
        ax.set_ylabel("body-frame y (up +)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Multi-window feature PCA on clean yolov8s data (k={K} clusters)",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "figure_combined_features.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {RESULTS_DIR / 'figure_combined_features.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
