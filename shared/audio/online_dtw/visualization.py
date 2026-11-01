"""Plotting helpers for the online tracker.

Two main figures are produced:

    1. ``alignment_curve``   - live time on x, reference time on y, with
       online OLTW path, offline-DTW pseudo-ground-truth, error band, and
       confidence shading.

    2. ``cost_surface``      - downsampled pairwise distance matrix with
       the online path overlaid (faithful but heavy; mainly for short
       segments and debugging).

Both use a single muted-but-readable matplotlib style with no emoji
glyphs and no decorative chrome.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_TRACKER_COLOR = "#d6336c"      # online OLTW
_GT_COLOR = "#1864ab"           # offline DTW pseudo-truth
_ERR_COLOR = "#ffa94d"          # error band
_CONF_COLOR = "#5c940d"
_BG = "#fafafa"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")


def alignment_figure(
    history: dict[str, np.ndarray],
    gt_live_t: np.ndarray | None = None,
    gt_ref_t: np.ndarray | None = None,
    metrics: dict[str, float] | None = None,
    title: str = "Online DTW alignment",
) -> plt.Figure:
    """Two-panel figure: alignment curve + per-frame error and confidence."""

    fig = plt.figure(figsize=(13, 7), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.18)
    ax_main = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1], sharex=ax_main)
    ax_conf = fig.add_subplot(gs[2], sharex=ax_main)

    live_t = history["live_time"]
    ref_t = history["ref_time"]
    conf = history["confidence"]

    if gt_live_t is not None and gt_ref_t is not None:
        ax_main.plot(gt_live_t, gt_ref_t, color=_GT_COLOR, lw=2.2,
                     alpha=0.85, label="offline DTW (pseudo-GT)")

    ax_main.plot(live_t, ref_t, color=_TRACKER_COLOR, lw=1.5,
                 label="online OLTW")
    ax_main.set_ylabel("Reference time (s)")
    ax_main.set_title(title, fontsize=13, loc="left")
    ax_main.legend(loc="lower right", frameon=False)
    _style(ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    if gt_live_t is not None and gt_ref_t is not None:
        gt_at_pred = np.interp(live_t, gt_live_t, gt_ref_t)
        err = ref_t - gt_at_pred
        ax_err.fill_between(live_t, 0, err, color=_ERR_COLOR, alpha=0.45,
                            label="online - offline")
        ax_err.plot(live_t, err, color=_ERR_COLOR, lw=0.9)
        ax_err.axhline(0, color="#333", lw=0.6)
        ax_err.set_ylabel("Error (s)")
        ax_err.legend(loc="upper right", frameon=False)
    _style(ax_err)
    plt.setp(ax_err.get_xticklabels(), visible=False)

    ax_conf.fill_between(live_t, 0, conf, color=_CONF_COLOR, alpha=0.5)
    ax_conf.plot(live_t, conf, color=_CONF_COLOR, lw=0.9)
    ax_conf.set_ylim(0, max(0.05, float(np.max(conf)) * 1.1) if conf.size else 1.0)
    ax_conf.set_xlabel("Live time (s)")
    ax_conf.set_ylabel("Confidence")
    _style(ax_conf)

    if metrics:
        text = (
            f"median |err| = {metrics.get('median_abs_err_s', 0):.3f} s\n"
            f"mean   |err| = {metrics.get('mean_abs_err_s', 0):.3f} s\n"
            f"p90    |err| = {metrics.get('p90_abs_err_s', 0):.3f} s\n"
            f"bias         = {metrics.get('bias_s', 0):+.3f} s\n"
            f"<= 250 ms    = {metrics.get('rate_250ms', 0)*100:.1f} %\n"
            f"<= 1000 ms   = {metrics.get('rate_1000ms', 0)*100:.1f} %"
        )
        ax_main.text(
            0.012, 0.98, text, transform=ax_main.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#ccc", alpha=0.9),
        )
    return fig


def cost_surface_figure(
    live_feats: np.ndarray, ref_feats: np.ndarray,
    history: dict[str, np.ndarray], hop_sec: float,
    gt_live_t: np.ndarray | None = None, gt_ref_t: np.ndarray | None = None,
    metric: str = "cosine", max_size: int = 1200, ref_pad_sec: float = 30.0,
    title: str = "Pairwise distance + online path",
) -> plt.Figure:
    """Plot the downsampled pairwise-distance surface with the online path.

    Both axes are *downsampled by stride* so that even hour-scale recordings
    fit in a single figure. The path is then projected through the same
    stride. ``ref_pad_sec`` controls how much extra reference around the
    matched range is shown so the path stays visible without a 37-minute
    reference dwarfing a 3-minute live signal.
    """

    A = live_feats if live_feats.shape[1] < live_feats.shape[0] else live_feats.T
    B = ref_feats if ref_feats.shape[1] < ref_feats.shape[0] else ref_feats.T

    # Crop the reference to the range the online path actually visited
    # (with a configurable padding) so the figure stays readable on long
    # references.
    ref_lo_sec = float(history["ref_time"].min()) if history["ref_time"].size else 0.0
    ref_hi_sec = float(history["ref_time"].max()) if history["ref_time"].size else B.shape[0] * hop_sec
    if gt_ref_t is not None and gt_ref_t.size:
        ref_lo_sec = min(ref_lo_sec, float(gt_ref_t.min()))
        ref_hi_sec = max(ref_hi_sec, float(gt_ref_t.max()))
    ref_lo = max(0, int((ref_lo_sec - ref_pad_sec) / hop_sec))
    ref_hi = min(B.shape[0], int((ref_hi_sec + ref_pad_sec) / hop_sec) + 1)
    B = B[ref_lo:ref_hi]
    ref_t0 = ref_lo * hop_sec

    sl = max(1, A.shape[0] // max_size)
    sr = max(1, B.shape[0] // max_size)
    A_d = A[::sl]
    B_d = B[::sr]

    if metric == "cosine":
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            A_n = A_d / (np.linalg.norm(A_d, axis=1, keepdims=True) + 1e-9)
            B_n = B_d / (np.linalg.norm(B_d, axis=1, keepdims=True) + 1e-9)
            D = 1.0 - A_n @ B_n.T
        D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
    else:
        a2 = np.sum(A_d * A_d, axis=1, keepdims=True)
        b2 = np.sum(B_d * B_d, axis=1, keepdims=True).T
        D = np.sqrt(np.maximum(a2 + b2 - 2.0 * (A_d @ B_d.T), 0.0))

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    extent = (ref_t0, ref_t0 + B_d.shape[0] * sr * hop_sec,
              0, A_d.shape[0] * sl * hop_sec)
    ax.imshow(D, origin="lower", aspect="auto", extent=extent, cmap="magma_r")
    if gt_live_t is not None and gt_ref_t is not None:
        ax.plot(gt_ref_t, gt_live_t, color=_GT_COLOR, lw=1.6,
                alpha=0.85, label="offline DTW (pseudo-GT)")
    ax.plot(history["ref_time"], history["live_time"],
            color=_TRACKER_COLOR, lw=1.4, label="online OLTW")
    ax.set_xlabel("Reference time (s)")
    ax.set_ylabel("Live time (s)")
    ax.set_title(title, fontsize=13, loc="left")
    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.85)
    for text in leg.get_texts():
        text.set_color("#222")
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 140) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
