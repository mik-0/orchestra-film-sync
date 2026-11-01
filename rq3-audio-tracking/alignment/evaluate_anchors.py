"""Benchmark the online tracker against hand-annotated ground truth.

Reads ``output/anchors.json`` (produced by ``scripts/annotate.py``), runs
the tracker over the same audio, and reports error at each anchor plus
overall summary statistics. Linear interpolation between anchors gives a
piecewise-linear ground-truth mapping ``live_t -> ref_t``; the tracker's
``ref_t`` is compared to this at every emitted frame.

Default behaviour cuts both audio files at the END of the first movement
(passed via ``--end-live`` and ``--end-ref``) so applause / silence at
movement boundaries do not pollute the metric.

Output:

    * a console table of per-anchor errors,
    * summary metrics (median, mean, p90, % within thresholds),
    * a PNG with two panels: (top) live_t -> ref_t with online OLTW,
      offline-DTW pseudo-GT, and the manual anchors all overlaid;
      (bottom) per-frame tracker error vs the anchor-GT.

Use this to find segments where the tracker is silently wrong.
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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from online_dtw import OnlineTracker, FeatureConfig, TrackerConfig
from online_dtw.oltw import OLTWConfig
from online_dtw.audio_io import load_mono
from online_dtw.features import extract_offline
from online_dtw.evaluation import (
    offline_reference_path, gt_function_from_path,
)


HERE = _AUDIO_ROOT
DEFAULT_LIVE = HERE / "Yuja Wang_ Rachmaninov Piano Concerto No. 2 in C minor Op. 18 [HD].mp3"
DEFAULT_REF = HERE / "Rachmaninoff_ Piano Concerto no.2 op.18 - Anna Fedorova - Complete Live Concert - HD.mp3"
DEFAULT_ANCHORS = HERE / "output" / "anchors.json"


# ---------------------------------------------------------------- helpers

def fmt_time(t: float) -> str:
    if t is None or not np.isfinite(t):
        return "  --:--"
    sign = "-" if t < 0 else " "
    t = abs(t)
    m, s = divmod(t, 60.0)
    return f"{sign}{int(m):02d}:{s:06.3f}"


def gt_from_anchors(anchors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(live_t, ref_t)`` arrays sorted by live_t for np.interp."""

    a = sorted(anchors, key=lambda a: a["live_t"])
    return (
        np.asarray([x["live_t"] for x in a], dtype=np.float64),
        np.asarray([x["ref_t"]  for x in a], dtype=np.float64),
    )


def interp_gt(live_t: np.ndarray, anchor_lt: np.ndarray, anchor_rt: np.ndarray,
              ) -> np.ndarray:
    """Piecewise-linear interpolation, masking points outside the anchor span."""

    out = np.full_like(live_t, np.nan, dtype=np.float64)
    if anchor_lt.size == 0:
        return out
    inside = (live_t >= anchor_lt[0]) & (live_t <= anchor_lt[-1])
    out[inside] = np.interp(live_t[inside], anchor_lt, anchor_rt)
    return out


# ---------------------------------------------------------------- runner

def _reanchor_subseq_dtw(
    live_buf: np.ndarray, ref_feats: np.ndarray, current_j: int,
    hop_sec: float, ref_search_sec: float, snap_threshold_sec: float,
    metric: str = "cosine",
) -> int | None:
    """Run subsequence DTW of a recent live buffer against a windowed slice
    of the reference around ``current_j``. Returns a snap target ``new_j``
    (in global ref frames) iff the offline DTW disagrees with ``current_j``
    by more than ``snap_threshold_sec``; otherwise ``None``.

    Why this works: offline DTW sees the whole live buffer forward AND
    backward, so it's much harder to derail than the online HMM. A periodic
    "is the online tracker still on the diagonal?" sanity check against
    the offline path is the Brazier-style hard re-anchor adapted for
    cumulative drift on long-form orchestral material.
    """
    import librosa

    n_ref_half = int(ref_search_sec / hop_sec)
    ref_lo = max(0, current_j - n_ref_half)
    ref_hi = min(ref_feats.shape[0], current_j + n_ref_half)
    ref_slice = ref_feats[ref_lo:ref_hi]
    if ref_slice.shape[0] < 5 or live_buf.shape[0] < 5:
        return None

    # librosa expects (n_features, n_frames). NaN-scrub mirrors the
    # auto_anchor.py logic; silent frames produce zero-norm chroma which
    # blows up cosine in DTW.
    X = live_buf.T.astype(np.float32)
    Y = ref_slice.T.astype(np.float32)
    for M in (X, Y):
        norms = np.linalg.norm(np.nan_to_num(M, nan=0.0), axis=0)
        bad = np.isnan(M).any(axis=0) | (norms < 1e-9)
        if bad.any():
            M[:, bad] = 0.0
            M[0, bad] = 1.0       # any unit vector

    try:
        _, wp = librosa.sequence.dtw(X=X, Y=Y, metric=metric, subseq=True)
    except Exception:
        return None
    # wp is [(live_idx, ref_idx_in_slice), ...] returned end-to-start
    end_ref_in_slice = int(wp[0, 1])
    new_j_global = ref_lo + end_ref_in_slice

    drift_sec = (current_j - new_j_global) * hop_sec
    if abs(drift_sec) > snap_threshold_sec:
        return int(new_j_global)
    return None


def _snap_hmm(hmm, new_j: int) -> None:
    """Snap the HMM forward variable to a new ref position without
    losing the Kalman tempo state. Used by --reanchor-interval to
    correct cumulative drift without wiping accumulated tempo learning.
    """
    M = hmm.M
    new_j = max(0, min(M - 1, int(new_j)))
    hmm._D_prev[:] = np.inf
    hmm._D_prev[new_j] = 0.0
    # Soft surround so the path can settle into nearby cells without
    # immediately jumping back to the old j_hat region.
    for k in range(1, 21):
        if new_j - k >= 0:
            hmm._D_prev[new_j - k] = float(k) * 0.5
        if new_j + k < M:
            hmm._D_prev[new_j + k] = float(k) * 0.5
    hmm.j_hat = new_j


def run_tracker(
    *, live_audio: np.ndarray, ref_audio: np.ndarray, sr: int, hop: int,
    feature: str, oltw_overrides: dict, start_j: int,
    backend: str = "oltw", beta: float | None = None,
    skip_mass: float | None = None, distance: str | None = None,
    emission_buffer_frames: int | None = None,
    offline_features: bool = False,
    reanchor_interval: float = 0.0,
    reanchor_live_window: float = 30.0,
    reanchor_ref_search: float = 60.0,
    reanchor_snap_threshold: float = 5.0,
) -> dict:
    feat_cfg = FeatureConfig(sr=sr, n_fft=4096, hop_length=hop, feature=feature)
    base = OLTWConfig(start_j=start_j)
    oltw_cfg = OLTWConfig(**{**base.__dict__, **oltw_overrides})
    from online_dtw.hmm import HMMConfig
    hmm_kwargs = dict(
        start_j=start_j,
        max_advance=oltw_overrides.get("max_advance", 2),
        search_width=oltw_overrides.get("search_width", 400),
        backward_penalty=oltw_overrides.get("backward_penalty", 0.001),
    )
    if beta is not None:
        hmm_kwargs["beta"] = beta
    if skip_mass is not None:
        hmm_kwargs["skip_mass"] = skip_mass
    if distance is not None:
        hmm_kwargs["distance"] = distance
    if emission_buffer_frames is not None:
        hmm_kwargs["emission_buffer_frames"] = emission_buffer_frames
    hmm_cfg = HMMConfig(**hmm_kwargs)
    cfg = TrackerConfig(feature=feat_cfg, oltw=oltw_cfg, hmm=hmm_cfg,
                         backend=backend)

    if offline_features:
        # Pre-extract both sides ONCE via extract_offline, then drive the
        # inner HMM (or OLTW) step-by-step. Avoids the streaming
        # extractor path entirely. Required for MERT-class features
        # whose streaming wrapper is O(N^2) and produces inconsistent
        # boundary-truncated outputs for early frames.
        from online_dtw.features import extract_offline as _ex_off
        from online_dtw.hmm import HMMTracker
        from online_dtw.oltw import OLTW
        print(f"  pre-extracting {feature} features offline (live + ref)...")
        live_feats = _ex_off(live_audio, feat_cfg).T          # (T_live, D)
        ref_feats  = _ex_off(ref_audio,  feat_cfg).T          # (T_ref,  D)
        print(f"    live: {live_feats.shape}   ref: {ref_feats.shape}")

        if backend == "hmm":
            inner = HMMTracker(ref_feats, hmm_cfg)
        else:
            inner = OLTW(ref_feats, oltw_cfg)
        live_times: list[float] = []
        ref_times: list[float] = []
        confs: list[float] = []
        hop_sec = feat_cfg.hop_sec

        reanchor_frames = int(reanchor_interval / hop_sec) if reanchor_interval > 0 else 0
        live_window_frames = int(reanchor_live_window / hop_sec)
        snaps: list[tuple[float, float, float]] = []   # (live_t, old_ref, new_ref)

        for n, lf in enumerate(live_feats):
            try:
                j, c = inner.step(lf, force_advance=None)
            except TypeError:
                j, c = inner.step(lf)

            # Periodic re-anchor: every reanchor_frames steps (skipping
            # the very first interval so the tracker has settled).
            if (reanchor_frames > 0 and backend == "hmm"
                    and n >= reanchor_frames
                    and n % reanchor_frames == 0):
                start_l = max(0, n - live_window_frames)
                live_buf = live_feats[start_l:n + 1]
                metric = hmm_cfg.distance if hmm_cfg.distance in ("cosine", "l1", "euclidean") else "cosine"
                new_j = _reanchor_subseq_dtw(
                    live_buf, ref_feats, current_j=j,
                    hop_sec=hop_sec, ref_search_sec=reanchor_ref_search,
                    snap_threshold_sec=reanchor_snap_threshold,
                    metric=metric,
                )
                if new_j is not None:
                    snaps.append((n * hop_sec, j * hop_sec, new_j * hop_sec))
                    _snap_hmm(inner, new_j)
                    j = new_j

            live_times.append(n * hop_sec)
            ref_times.append(j * hop_sec)
            confs.append(float(c))

        if snaps:
            print(f"  re-anchor fired {len(snaps)} times:")
            for live_t, old, new in snaps[:8]:
                print(f"    @ {live_t:7.1f} s   {old:7.1f} -> {new:7.1f}   (Δ={new-old:+.1f} s)")
            if len(snaps) > 8:
                print(f"    ... and {len(snaps) - 8} more")

        return {
            "live_time": np.asarray(live_times),
            "ref_time":  np.asarray(ref_times),
            "confidence": np.asarray(confs),
            "reanchors": snaps,
            "feat_cfg": feat_cfg, "oltw_cfg": oltw_cfg,
        }

    tracker = OnlineTracker(ref_audio, cfg)
    chunk_n = int(0.10 * sr)
    for k in range(0, len(live_audio), chunk_n):
        list(tracker.update(live_audio[k:k + chunk_n]))
    h = tracker.history()
    return {
        "live_time": h["live_time"], "ref_time": h["ref_time"],
        "confidence": h["confidence"], "feat_cfg": feat_cfg, "oltw_cfg": oltw_cfg,
        "reanchors": [],
    }


def offline_pseudo_gt(
    *, live_audio: np.ndarray, ref_audio: np.ndarray, feat_cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    A = extract_offline(live_audio, feat_cfg)
    B = extract_offline(ref_audio, feat_cfg)
    li, ri = offline_reference_path(A, B, metric="cosine", subseq=True)
    return gt_function_from_path(li, ri, hop_sec=feat_cfg.hop_sec)


# ---------------------------------------------------------------- metrics

def metrics_against_anchors(
    pred_live_t: np.ndarray, pred_ref_t: np.ndarray,
    anchor_lt: np.ndarray, anchor_rt: np.ndarray,
) -> dict:
    """Compute per-frame metrics over the anchor-spanned region."""

    gt_at_pred = interp_gt(pred_live_t, anchor_lt, anchor_rt)
    mask = np.isfinite(gt_at_pred)
    err = pred_ref_t[mask] - gt_at_pred[mask]
    abs_err = np.abs(err)
    if abs_err.size == 0:
        return {"n_frames": 0}
    return {
        "n_frames": int(abs_err.size),
        "median_abs_err_s": float(np.median(abs_err)),
        "mean_abs_err_s":   float(np.mean(abs_err)),
        "p90_abs_err_s":    float(np.percentile(abs_err, 90)),
        "max_abs_err_s":    float(np.max(abs_err)),
        "bias_s":           float(np.median(err)),
        "rate_100ms":       float(np.mean(abs_err <= 0.100)),
        "rate_250ms":       float(np.mean(abs_err <= 0.250)),
        "rate_500ms":       float(np.mean(abs_err <= 0.500)),
        "rate_1000ms":      float(np.mean(abs_err <= 1.000)),
        "rate_5s":          float(np.mean(abs_err <= 5.000)),
    }


def per_anchor_errors(
    pred_live_t: np.ndarray, pred_ref_t: np.ndarray,
    anchor_lt: np.ndarray, anchor_rt: np.ndarray,
) -> list[float | None]:
    """Tracker's predicted ref minus anchor-GT ref at each anchor's live_t."""

    out: list[float | None] = []
    for lt, rt in zip(anchor_lt, anchor_rt):
        if pred_live_t.size == 0 or lt < pred_live_t[0] or lt > pred_live_t[-1]:
            out.append(None)
            continue
        pred_at = float(np.interp(lt, pred_live_t, pred_ref_t))
        out.append(pred_at - float(rt))
    return out


# ---------------------------------------------------------------- plotting

def plot_results(
    *, anchor_lt: np.ndarray, anchor_rt: np.ndarray,
    online_lt: np.ndarray, online_rt: np.ndarray,
    offline_lt: np.ndarray | None, offline_rt: np.ndarray | None,
    out_path: Path, title: str,
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 8), facecolor="#0e1117")
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.3,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#1a1f2b")
        for s in ax.spines.values():
            s.set_color("#7a8290"); s.set_linewidth(0.5)
        ax.tick_params(colors="#7a8290", labelsize=9)
        ax.grid(True, color="#7a8290", alpha=0.18, lw=0.5, ls=":")

    if offline_lt is not None and offline_lt.size:
        ax_top.plot(offline_lt, offline_rt, color="#5fc4ff", lw=1.6,
                     alpha=0.65, label="offline DTW (pseudo-GT)")
    ax_top.plot(online_lt, online_rt, color="#ff6b9d", lw=1.4,
                 alpha=0.95, label="online OLTW")
    # Anchor segments + dots
    ax_top.plot(anchor_lt, anchor_rt, color="#ffd166",
                 lw=1.4, alpha=0.9, label="hand anchors (linear interp)")
    ax_top.plot(anchor_lt, anchor_rt, "x", color="#ffd166", ms=12, mew=2)
    ax_top.set_ylabel("Reference time (s)", color="#e8e8ee")
    ax_top.set_title(title, color="#e8e8ee", fontsize=12, loc="left")
    leg = ax_top.legend(loc="upper left", frameon=True, framealpha=0.85)
    for t in leg.get_texts(): t.set_color("#e8e8ee")

    # Per-frame error against anchor-GT (only inside the anchor span)
    gt_at_online = interp_gt(online_lt, anchor_lt, anchor_rt)
    err = online_rt - gt_at_online
    mask = np.isfinite(err)
    ax_bot.fill_between(online_lt[mask], 0, err[mask],
                         color="#ffa94d", alpha=0.45)
    ax_bot.plot(online_lt[mask], err[mask], color="#ffa94d", lw=0.8)
    ax_bot.axhline(0, color="#7a8290", lw=0.6)
    ax_bot.set_xlabel("Live time (s)", color="#e8e8ee")
    ax_bot.set_ylabel("online - GT (s)", color="#e8e8ee")

    # Vertical dashed lines at each anchor for reference
    for lt in anchor_lt:
        ax_top.axvline(lt, color="#ffd166", lw=0.5, alpha=0.25)
        ax_bot.axvline(lt, color="#ffd166", lw=0.5, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="#0e1117", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    p.add_argument("--ref", type=Path, default=DEFAULT_REF)
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop", type=int, default=1024)
    p.add_argument("--feature", default="chroma")
    p.add_argument("--end-live", type=float, default=645.027,
                   help="cut live audio at this many seconds (e.g. end of "
                        "movement 1)")
    p.add_argument("--end-ref", type=float, default=678.175,
                   help="cut reference audio at this many seconds")
    p.add_argument("--seed", choices=("naive", "anchor"), default="anchor",
                   help="how to seed the OLTW: 'naive' uses 1:1 (start_j=0); "
                        "'anchor' seeds at the first anchor's ref_t")
    p.add_argument("--max-advance", type=int, default=2)
    p.add_argument("--search-width", type=int, default=400)
    p.add_argument("--backend", choices=("oltw", "hmm"), default="oltw")
    p.add_argument("--beta", type=float, default=None,
                   help="HMM emission temperature (overrides HMMConfig "
                        "default of 8.0). Cross-recording presets use 32.")
    p.add_argument("--skip-mass", type=float, default=None,
                   help="HMM global skip-mass for absorbing-state escape "
                        "(overrides HMMConfig default of 1e-5). Pass 0 "
                        "to disable and reproduce pre-fix behaviour.")
    p.add_argument("--emission-buffer-frames", type=int, default=None,
                   help="HMM emission buffer (Otsuka audio-buffer matching). "
                        "Average per-frame distances over this many recent "
                        "live frames before computing the HMM emission. 0 "
                        "(default in code) = single-frame; 20-50 frames "
                        "(~1-2.3 s at hop=1024/sr=22050) suppresses per-frame "
                        "noise that compounds into long-form drift.")
    p.add_argument("--distance", choices=("cosine", "l1", "euclidean"),
                   default=None,
                   help="HMM emission distance. Default cosine; matchmaker's "
                        "audio default is L1 (Manhattan). L1 sums absolute "
                        "differences across feature bins, which tends to be "
                        "more forgiving of single-bin spikes than cosine's "
                        "geometric similarity.")
    p.add_argument("--offline-features", action="store_true",
                   help="Pre-extract live + ref features via extract_offline "
                        "once and feed them directly to the inner tracker. "
                        "Required for MERT-class features whose streaming "
                        "extractor is currently O(N^2) and produces "
                        "boundary-truncated early frames. Strictly equivalent "
                        "to the streaming path for chroma / LSE since their "
                        "streaming extractor matches extract_offline frame-for-frame.")
    p.add_argument("--reanchor-interval", type=float, default=0.0,
                   help="seconds between subseq-DTW re-anchor checks; 0 disables. "
                        "Only effective with --offline-features and --backend hmm.")
    p.add_argument("--reanchor-live-window", type=float, default=30.0,
                   help="seconds of recent live features used for the re-anchor "
                        "subseq-DTW.")
    p.add_argument("--reanchor-ref-search", type=float, default=60.0,
                   help="seconds of ref to search on each side of the tracker's "
                        "current j_hat during a re-anchor check.")
    p.add_argument("--reanchor-snap-threshold", type=float, default=5.0,
                   help="snap to the offline-DTW position only if it disagrees "
                        "with the online tracker by more than this many seconds.")
    p.add_argument("--no-offline", action="store_true",
                   help="skip the offline-DTW pseudo-GT comparison (faster)")
    p.add_argument("--out-png", type=Path,
                   default=HERE / "output" / "anchor_eval.png")
    p.add_argument("--out-json", type=Path,
                   default=HERE / "output" / "anchor_eval.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("loading audio (cached)...")
    live_full, _ = load_mono(args.live, target_sr=args.sr)
    ref_full, _ = load_mono(args.ref, target_sr=args.sr)
    live_audio = live_full[: int(args.end_live * args.sr)]
    ref_audio  = ref_full[:  int(args.end_ref  * args.sr)]
    print(f"  cut to live={args.end_live:.1f}s, ref={args.end_ref:.1f}s "
          f"(end of first movement)")

    with open(args.anchors) as f:
        payload = json.load(f)
    anchor_lt, anchor_rt = gt_from_anchors(payload["anchors"])
    # Drop anchors past the cut
    keep = (anchor_lt <= args.end_live) & (anchor_rt <= args.end_ref)
    anchor_lt = anchor_lt[keep]; anchor_rt = anchor_rt[keep]
    print(f"  using {anchor_lt.size} anchors over "
          f"[{anchor_lt.min():.1f}, {anchor_lt.max():.1f}] s of live")

    # Seed the tracker. With "anchor" seeding, start at the first anchor's
    # ref_t so the eval measures *tracking* (not localization).
    feat_cfg = FeatureConfig(sr=args.sr, n_fft=4096, hop_length=args.hop,
                              feature=args.feature)
    if args.seed == "anchor":
        start_j = int(anchor_rt[0] / feat_cfg.hop_sec)
        seed_lbl = f"anchor[0]={anchor_rt[0]:.2f}s"
    else:
        start_j = 0
        seed_lbl = "naive (start_j=0)"
    print(f"  seeding OLTW: {seed_lbl}")

    print("running online tracker...")
    t0 = time.perf_counter()
    res = run_tracker(
        live_audio=live_audio, ref_audio=ref_audio, sr=args.sr, hop=args.hop,
        feature=args.feature, start_j=start_j, backend=args.backend,
        beta=args.beta, skip_mass=args.skip_mass, distance=args.distance,
        emission_buffer_frames=args.emission_buffer_frames,
        offline_features=args.offline_features,
        reanchor_interval=args.reanchor_interval,
        reanchor_live_window=args.reanchor_live_window,
        reanchor_ref_search=args.reanchor_ref_search,
        reanchor_snap_threshold=args.reanchor_snap_threshold,
        oltw_overrides={"max_advance": args.max_advance,
                        "search_width": args.search_width},
    )
    print(f"  online run: {time.perf_counter() - t0:.1f} s "
          f"({res['live_time'].size} frames)")

    online_lt = res["live_time"]
    online_rt = res["ref_time"]
    # The first anchor isn't at t=0; if seeding at anchor[0] live_t > 0,
    # offset the tracker time so live_t matches the actual live clock.
    if args.seed == "anchor":
        online_lt = online_lt + anchor_lt[0]

    # Offline pseudo-GT for visual comparison
    offline_lt = offline_rt = None
    if not args.no_offline:
        print("computing offline-DTW pseudo-GT for visual comparison...")
        t0 = time.perf_counter()
        offline_lt, offline_rt = offline_pseudo_gt(
            live_audio=live_audio, ref_audio=ref_audio, feat_cfg=feat_cfg,
        )
        print(f"  offline DTW: {time.perf_counter() - t0:.1f} s")

    # Per-anchor error
    per_anchor = per_anchor_errors(online_lt, online_rt, anchor_lt, anchor_rt)
    if not args.no_offline:
        per_anchor_offline = per_anchor_errors(
            offline_lt, offline_rt, anchor_lt, anchor_rt,
        )
    else:
        per_anchor_offline = [None] * len(anchor_lt)

    # Summary
    metrics = metrics_against_anchors(online_lt, online_rt, anchor_lt, anchor_rt)
    print("\n=== per-anchor error (online_ref - GT_ref) ===")
    print(f"{'live':>10s}  {'ref(GT)':>10s}  {'online_err':>11s}  {'offline_err':>11s}")
    print("-" * 50)
    for i in range(len(anchor_lt)):
        on = per_anchor[i]
        of = per_anchor_offline[i]
        on_s = f"{on:+8.3f} s" if on is not None else "    -- "
        of_s = f"{of:+8.3f} s" if of is not None else "    -- "
        print(f"{fmt_time(anchor_lt[i])}  {fmt_time(anchor_rt[i])}  "
              f"{on_s:>11s}  {of_s:>11s}")

    print("\n=== summary metrics (online OLTW vs anchor-GT) ===")
    if metrics["n_frames"] == 0:
        print("  no overlap with anchor span")
    else:
        for k, v in metrics.items():
            if k == "n_frames":
                print(f"  {k:<20s} {int(v):>8d}")
            elif "rate" in k:
                print(f"  {k:<20s} {v*100:>7.2f} %")
            else:
                print(f"  {k:<20s} {v:>+8.3f} s")

    # Save JSON + PNG
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "config": {
                "feature": args.feature, "hop": args.hop,
                "max_advance": args.max_advance,
                "search_width": args.search_width,
                "seed": args.seed,
            },
            "metrics": metrics,
            "per_anchor": [
                {"live_t": float(anchor_lt[i]), "ref_t": float(anchor_rt[i]),
                 "online_err": per_anchor[i],
                 "offline_err": per_anchor_offline[i]}
                for i in range(len(anchor_lt))
            ],
        }, f, indent=2)
    print(f"\n  metrics written to {args.out_json}")

    plot_results(
        anchor_lt=anchor_lt, anchor_rt=anchor_rt,
        online_lt=online_lt, online_rt=online_rt,
        offline_lt=offline_lt, offline_rt=offline_rt,
        out_path=args.out_png,
        title=(f"online OLTW vs hand anchors  --  feature={args.feature}, "
               f"max_adv={args.max_advance}, c={args.search_width}, "
               f"seed={args.seed}"),
    )
    print(f"  plot written to {args.out_png}")


if __name__ == "__main__":
    main()
