"""Evaluation utilities: pseudo-ground-truth via offline DTW + metrics.

Strict ground truth for cross-performance audio-to-audio alignment is
expensive (note-level annotations on both recordings). For automated
benchmarking we use the standard fallback: run a non-causal global DTW
across the full feature matrices and treat its warping path as
"pseudo-ground-truth". This is **not** a perfect oracle (it is biased by
the same feature representation) but it is a strong baseline against
which the online tracker's lag can be measured fairly.

Reported metrics:

    median_abs_err_s   - median |t_pred - t_gt| in seconds
    mean_abs_err_s     - mean   |t_pred - t_gt| in seconds
    p90_abs_err_s      - 90th-percentile |t_pred - t_gt|
    bias_s             - signed median(t_pred - t_gt). Positive means the
                         tracker is *ahead* of the offline alignment.
    rate_within_xxx_ms - fraction of frames within {100, 250, 500, 1000} ms.
    coverage           - fraction of live frames we have any prediction for.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import librosa


@dataclass(frozen=True)
class AlignmentMetrics:
    median_abs_err_s: float
    mean_abs_err_s: float
    p90_abs_err_s: float
    bias_s: float
    rate_100ms: float
    rate_250ms: float
    rate_500ms: float
    rate_1000ms: float
    coverage: float
    n_frames: int

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__annotations__}


def offline_reference_path(
    live_feats: np.ndarray, ref_feats: np.ndarray, *,
    metric: str = "cosine", subseq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run librosa global (or subsequence) DTW on the feature matrices.

    Inputs are passed through as-is to librosa, which expects shape
    ``(n_features, n_frames)``. We auto-orient so callers can pass either
    ``(F, T)`` or ``(T, F)`` and get the right answer.

    When ``subseq=True`` the live signal is matched as a subsequence of
    the reference: the start and end of ``X`` align *somewhere* inside
    ``Y`` rather than to its endpoints. This is the right call when the
    reference is significantly longer than the live signal (e.g. when
    truncating the live for a quick benchmark).

    Returns two ascending-by-live arrays ``(live_idx, ref_idx)``.
    """

    X = live_feats if live_feats.shape[0] < live_feats.shape[1] else live_feats.T
    Y = ref_feats if ref_feats.shape[0] < ref_feats.shape[1] else ref_feats.T

    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric=metric, subseq=subseq)
    wp = wp[::-1]  # librosa returns end-to-start
    live_idx = wp[:, 0]
    ref_idx = wp[:, 1]
    return live_idx, ref_idx


def gt_function_from_path(
    live_idx: np.ndarray, ref_idx: np.ndarray, hop_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a step-shaped DTW path into a per-frame ground-truth array.

    For each unique live frame we report the *median* matched reference
    frame (handles the horizontal-step plateaus naturally). Returns:

        unique_live_t : float seconds
        gt_ref_t      : float seconds
    """

    order = np.argsort(live_idx, kind="stable")
    li = live_idx[order]
    ri = ref_idx[order]
    uniq, starts = np.unique(li, return_index=True)
    gt = np.empty(uniq.size, dtype=np.float64)
    starts = np.append(starts, len(li))
    for k in range(uniq.size):
        gt[k] = np.median(ri[starts[k] : starts[k + 1]])
    return uniq * hop_sec, gt * hop_sec


def metrics_from_paths(
    pred_live_t: np.ndarray, pred_ref_t: np.ndarray,
    gt_live_t: np.ndarray, gt_ref_t: np.ndarray,
    n_total: int | None = None,
) -> AlignmentMetrics:
    """Compute summary metrics by interpolating the GT onto the predicted grid."""

    # Interpolate GT at predicted live times.
    gt_at_pred = np.interp(pred_live_t, gt_live_t, gt_ref_t)
    err = pred_ref_t - gt_at_pred
    abs_err = np.abs(err)

    n = abs_err.size
    if n_total is None:
        n_total = n
    return AlignmentMetrics(
        median_abs_err_s=float(np.median(abs_err)),
        mean_abs_err_s=float(np.mean(abs_err)),
        p90_abs_err_s=float(np.percentile(abs_err, 90)),
        bias_s=float(np.median(err)),
        rate_100ms=float(np.mean(abs_err <= 0.100)),
        rate_250ms=float(np.mean(abs_err <= 0.250)),
        rate_500ms=float(np.mean(abs_err <= 0.500)),
        rate_1000ms=float(np.mean(abs_err <= 1.000)),
        coverage=float(n / n_total),
        n_frames=int(n),
    )
