"""Paired moving-block resampling for the audio-context correlations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class RankBootstrapResult:
    rho: float | None
    lower: float | None
    upper: float | None
    defined_resamples: int
    undefined_resamples: int


def moving_block_rank_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
    legacy_fixed_ranks: bool = False,
) -> RankBootstrapResult:
    """Bootstrap Spearman correlation, reranking within every resample.

    Inputs must be finite, paired observations on a regular time grid. The
    caller validates that grid; dropping interior missing observations would
    change the temporal meaning of the blocks. Blocks overlap, are sampled
    with replacement, and the final concatenated block is truncated to n.

    ``legacy_fixed_ranks`` reproduces the original calculation solely for
    auditing preserved results. Undefined correlations are counted explicitly;
    percentile bounds use defined resamples and are absent if none are defined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("expected paired one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite observations require an explicit gap policy")
    if repetitions < 1 or block_size < 1:
        raise ValueError("repetitions and block_size must be positive")
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return RankBootstrapResult(None, None, None, 0, repetitions)

    x_rank, y_rank = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    block_size = min(block_size, x.size)
    starts = np.arange(x.size - block_size + 1)
    offsets = np.arange(block_size)
    blocks_needed = int(np.ceil(x.size / block_size))
    rng = np.random.default_rng(seed)
    estimates = np.full(repetitions, np.nan)
    for index in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = (chosen[:, None] + offsets).ravel()[:x.size]
        a, b = x_rank[sample], y_rank[sample]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        if not legacy_fixed_ranks:
            # Ranking the original ranks is equivalent to ranking the values,
            # including ties, and avoids any rounding of the original values.
            a, b = rankdata(a), rankdata(b)
        estimates[index] = np.corrcoef(a, b)[0, 1]
    defined = estimates[np.isfinite(estimates)]
    bounds = np.percentile(defined, [2.5, 97.5]) if defined.size else (None, None)
    return RankBootstrapResult(
        rho,
        float(bounds[0]) if defined.size else None,
        float(bounds[1]) if defined.size else None,
        int(defined.size),
        repetitions - int(defined.size),
    )
