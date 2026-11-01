"""Utilities for audited audio-alignment reference paths.

The functions in this module deliberately do not depend on Sync Toolbox.  They
operate on a monotonic mapping in seconds, which keeps structural quality
control and failure-episode extraction testable without optional dependencies.
"""

from __future__ import annotations

import numpy as np


def sample_mapping(
    live_t: np.ndarray,
    ref_t: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` evenly spaced correspondences from a dense mapping."""

    if count < 2:
        raise ValueError("count must be at least 2")
    if live_t.size < 2 or ref_t.size != live_t.size:
        raise ValueError("alignment path must contain matching live/ref arrays")
    target_live = np.linspace(float(live_t[0]), float(live_t[-1]), count)
    target_ref = np.interp(target_live, live_t, ref_t)
    return target_live, target_ref


def structural_qc(
    live_t: np.ndarray,
    ref_t: np.ndarray,
    *,
    window_seconds: float = 10.0,
    min_slope: float = 0.25,
    max_slope: float = 4.0,
) -> dict:
    """Flag path regions whose local tempo ratio indicates a structural error.

    The dense path is resampled on a fixed live-time grid.  ``slope`` is the
    reference-time advance divided by live-time advance over one QC window.
    Very small values are stalls; very large values are jumps.  The defaults
    are intentionally permissive and catch forced endpoint matches rather than
    ordinary expressive tempo variation.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0 <= min_slope < max_slope:
        raise ValueError("expected 0 <= min_slope < max_slope")
    if live_t.size < 2 or ref_t.size != live_t.size:
        raise ValueError("alignment path must contain matching live/ref arrays")

    start = float(live_t[0])
    end = float(live_t[-1])
    grid = np.arange(start, end, window_seconds, dtype=np.float64)
    if grid.size == 0 or grid[0] != start:
        grid = np.insert(grid, 0, start)
    if end - grid[-1] < 0.5 * window_seconds and grid.size > 1:
        grid[-1] = end
    elif grid[-1] < end:
        grid = np.append(grid, end)
    mapped = np.interp(grid, live_t, ref_t)
    live_delta = np.diff(grid)
    ref_delta = np.diff(mapped)
    slopes = np.divide(
        ref_delta,
        live_delta,
        out=np.full_like(ref_delta, np.nan),
        where=live_delta > 0,
    )

    flags = []
    for index, slope in enumerate(slopes):
        kind = None
        if slope < min_slope:
            kind = "stall"
        elif slope > max_slope:
            kind = "jump"
        if kind is not None:
            flags.append({
                "kind": kind,
                "live_start": round(float(grid[index]), 4),
                "live_end": round(float(grid[index + 1]), 4),
                "ref_start": round(float(mapped[index]), 4),
                "ref_end": round(float(mapped[index + 1]), 4),
                "slope": round(float(slope), 4),
            })

    finite_slopes = slopes[np.isfinite(slopes)]
    return {
        "passed": not flags,
        "window_seconds": window_seconds,
        "min_slope": min_slope,
        "max_slope": max_slope,
        "median_slope": (
            round(float(np.median(finite_slopes)), 4)
            if finite_slopes.size else None
        ),
        "min_observed_slope": (
            round(float(np.min(finite_slopes)), 4)
            if finite_slopes.size else None
        ),
        "max_observed_slope": (
            round(float(np.max(finite_slopes)), 4)
            if finite_slopes.size else None
        ),
        "flags": flags,
    }


def failure_episodes(
    live_t: np.ndarray,
    error_s: np.ndarray,
    *,
    threshold_s: float = 1.0,
    minimum_duration_s: float = 1.0,
) -> list[dict]:
    """Return contiguous episodes where absolute tracking error is large.

    ``live_t`` must be increasing and regularly or irregularly sampled.  An
    episode duration is measured from its first failing sample to the first
    following non-failing sample, or to the final sample for a trailing episode.
    """

    if live_t.size != error_s.size:
        raise ValueError("live_t and error_s must have the same length")
    if threshold_s <= 0 or minimum_duration_s < 0:
        raise ValueError("threshold must be positive and duration non-negative")
    if live_t.size == 0:
        return []
    if np.any(np.diff(live_t) <= 0):
        raise ValueError("live_t must be strictly increasing")

    failing = np.isfinite(error_s) & (np.abs(error_s) >= threshold_s)
    change = np.diff(np.r_[False, failing, False].astype(np.int8))
    starts = np.flatnonzero(change == 1)
    stops = np.flatnonzero(change == -1)
    episodes = []
    for start, stop in zip(starts, stops):
        end_t = float(live_t[stop]) if stop < live_t.size else float(live_t[-1])
        start_t = float(live_t[start])
        duration = max(0.0, end_t - start_t)
        if duration < minimum_duration_s:
            continue
        segment_error = error_s[start:stop]
        peak_index = start + int(np.nanargmax(np.abs(segment_error)))
        episodes.append({
            "live_start": round(start_t, 4),
            "live_end": round(end_t, 4),
            "duration_s": round(duration, 4),
            "peak_live_t": round(float(live_t[peak_index]), 4),
            "peak_error_s": round(float(error_s[peak_index]), 4),
            "median_abs_error_s": round(
                float(np.nanmedian(np.abs(segment_error))), 4
            ),
        })
    return episodes
