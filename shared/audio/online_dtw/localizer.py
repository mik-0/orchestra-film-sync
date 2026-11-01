"""Subsequence DTW startup localizer.

When the live performance does not begin at the start of the reference
(rehearsal restart, resumption after a pause, or simply unknown entry
point), naive OLTW initialized at ``j = 0`` will spend tens of seconds
catching up. This module runs a one-shot subsequence DTW from a short
initial buffer of live features against the full reference, and returns
the best-matching reference position to seed the OLTW tracker.

The implementation here is the standard subsequence DTW formulation:

    * Cost matrix C[i, j] = local_distance(A[i], B[j]) over the live
      buffer ``A`` (length T_live) and the reference ``B`` (length M).
    * D[0, j] = C[0, j]                  (free entry on the reference)
    * D[i, j] = C[i, j] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    * The localized end is ``argmin_j D[T_live - 1, j]``.

This is **not** itself a streaming aligner, but it is fully causal on the
*live* axis (we only see live frames 0..T_live-1) and operates only on a
short buffer, so calling it once at startup adds bounded latency at
session start and zero latency thereafter.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalizerConfig:
    distance: str = "cosine"


def _pairwise_distance(A: np.ndarray, B: np.ndarray, kind: str) -> np.ndarray:
    """Pairwise distance matrix between rows of A and rows of B."""

    if kind == "cosine":
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            D = 1.0 - A_n @ B_n.T
        return np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
    if kind == "euclidean":
        # broadcast-free squared distance via |a|^2 + |b|^2 - 2 a.b
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0))
    if kind == "l1":
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    raise ValueError(f"unknown distance: {kind}")


class SubsequenceLocalizer:
    """Startup-time localizer over a short live buffer."""

    def __init__(self, ref_features: np.ndarray, cfg: LocalizerConfig | None = None) -> None:
        self.B = np.ascontiguousarray(ref_features, dtype=np.float32)
        self.cfg = cfg or LocalizerConfig()

    def localize(self, live_buffer: np.ndarray) -> tuple[int, float]:
        """Return ``(j_start, score)``.

        ``live_buffer`` is shape ``(T_live, F)``. ``score`` is the cumulative
        cost at the matched end point, normalized by ``T_live`` so values
        are comparable across buffer lengths. Lower means better match.
        """

        if live_buffer.ndim != 2 or live_buffer.shape[1] != self.B.shape[1]:
            raise ValueError(
                f"live_buffer shape {live_buffer.shape} incompatible "
                f"with reference shape {self.B.shape}"
            )

        C = _pairwise_distance(live_buffer.astype(np.float32), self.B, self.cfg.distance)
        T, M = C.shape

        # D[0, :] = C[0, :]   (free start anywhere in the reference)
        D = np.full_like(C, np.inf, dtype=np.float64)
        D[0] = C[0]
        for i in range(1, T):
            # diag, vertical (ref stays), horizontal (live stays)
            shifted_diag = np.concatenate(([np.inf], D[i - 1, :-1]))
            shifted_horiz_above = D[i - 1]
            row = np.full(M, np.inf, dtype=np.float64)
            # we need row[j] depends on row[j-1], so do a left-to-right pass
            for j in range(M):
                best = shifted_horiz_above[j]
                if j > 0:
                    if shifted_diag[j] < best:
                        best = shifted_diag[j]
                    if row[j - 1] < best:
                        best = row[j - 1]
                row[j] = C[i, j] + best
            D[i] = row

        end_j = int(np.argmin(D[T - 1]))
        return end_j, float(D[T - 1, end_j] / T)
