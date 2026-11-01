"""Reproduce the deeper failure analysis for published Dixon traces.

This script deliberately does not modify thesis prose. It treats the audited
dense paths and frozen published-tracker traces as inputs, then writes one
self-contained result directory containing:

* ``summary.json``: protocols, provenance, descriptive and exploratory results;
* ``oracle_resets.csv``: one row per merged failure bout and reset lead; and
* ``README.md``: a compact human-readable report with interpretation limits.

The expensive stage extracts the exact Matchmaker 0.3.0 log-spectral features
from the source media. Run this script with the pinned Matchmaker environment
created by ``setup_published_baselines.py``.
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
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata


PROJECT_ROOT = _AUDIO_ROOT
REPOSITORY_ROOT = _REPO_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_audio_failures import moving_block_spearman_interval  # noqa: E402
from online_dtw.alignment_reference import failure_episodes  # noqa: E402
from online_dtw.presets import Preset  # noqa: E402
from run_published_tracker import (  # noqa: E402
    MATCHMAKER_COMMIT,
    MATCHMAKER_VERSION,
    compute_matchmaker_features_chunked,
    load_span_direct,
)


PAIR_SPECS = {
    "firebird": {
        "token": "firebird",
        "preset": "firebird",
        "reference": "firebird_reference.json",
    },
    "chopin_black_keys": {
        "token": "chopin",
        "preset": "chopin_black_keys",
        "reference": "chopin_black_keys_reference.json",
    },
    "yuja_fedorova": {
        "token": "yuja_fedorova",
        "preset": "yuja_fedorova",
        "reference": "yuja_fedorova_reference.json",
    },
    "daphnis_chloe": {
        "token": "daphnis",
        "preset": "daphnis_chloe",
        "reference": "daphnis_chloe_reference.json",
    },
}
METHOD_SUFFIXES = {
    "arzt": "matchmaker_arzt",
    "dixon": "matchmaker_dixon",
    "concertcue": "concertcue",
}
FRAME_RATE = 30
SAMPLE_RATE = 44100
HOP_SAMPLES = 1470
FFT_SAMPLES = 2 * HOP_SAMPLES
FAILURE_THRESHOLD_S = 1.0
MINIMUM_FAILURE_DURATION_S = 1.0
MERGE_GAP_S = 5.0
RESET_LEADS_S = (2.0, 5.0, 10.0, 20.0)
BOOTSTRAP_BLOCK_S = 30
BOOTSTRAP_REPETITIONS = 1000
RANDOM_SEED = 20260817


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "audio_failure_analysis",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the three generated files if they already exist",
    )
    return parser.parse_args()


def repository_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return path.name


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(predictor: np.ndarray, failed: np.ndarray) -> float | None:
    mask = np.isfinite(predictor)
    x = predictor[mask]
    y = failed[mask].astype(bool)
    positive = int(np.sum(y))
    negative = int(np.sum(~y))
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(x)
    rank_sum = float(np.sum(ranks[y]))
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def describe_predictor(predictor: np.ndarray, absolute_error: np.ndarray) -> dict:
    failed = absolute_error >= FAILURE_THRESHOLD_S
    rho, lower, upper = moving_block_spearman_interval(
        predictor,
        absolute_error,
        block_size=BOOTSTRAP_BLOCK_S,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=RANDOM_SEED,
    )
    return {
        "spearman_rho_abs_error": rho,
        "moving_block_bootstrap_95_interval": [lower, upper],
        "failure_auc": auc(predictor, failed),
        "median_during_failed_seconds": (
            float(np.nanmedian(predictor[failed])) if np.any(failed) else None
        ),
        "median_during_other_seconds": (
            float(np.nanmedian(predictor[~failed])) if np.any(~failed) else None
        ),
    }


def normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-10)


def compute_exact_features(path: Path, start: float, end: float) -> tuple[np.ndarray, float]:
    from matchmaker.features.audio import LogSpectralEnergyProcessor

    started = time.perf_counter()
    audio = load_span_direct(path, start=start, end=end, sr=SAMPLE_RATE)
    processor = LogSpectralEnergyProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_SAMPLES,
        normalize=False,
    )
    raw = compute_matchmaker_features_chunked(
        audio,
        processor=processor,
        hop=HOP_SAMPLES,
    )
    del audio
    return raw, time.perf_counter() - started


def run_dixon(reference_features: np.ndarray, live_features: np.ndarray) -> np.ndarray:
    from matchmaker.dp import OnlineTimeWarpingDixonFrame

    score_positions = np.arange(reference_features.shape[0], dtype=np.float32) / FRAME_RATE
    follower = OnlineTimeWarpingDixonFrame(
        reference_features=reference_features,
        score_positions=score_positions,
        ref_frame_to_beat=score_positions,
        frame_rate=FRAME_RATE,
    )
    positions = np.empty(live_features.shape[0], dtype=np.float64)
    for index, observation in enumerate(live_features):
        if follower.is_still_following():
            positions[index] = follower(observation, index / FRAME_RATE)
        else:
            positions[index:] = positions[index - 1] if index else 0.0
            break
    if hasattr(follower, "input_features"):
        follower.input_features.clear()
    if hasattr(follower, "_alignment_path"):
        follower._alignment_path.clear()
    return positions


def merge_failure_bouts(
    live_t: np.ndarray,
    error_s: np.ndarray,
    episodes: list[dict],
    *,
    maximum_gap_s: float = MERGE_GAP_S,
) -> list[dict]:
    if not episodes:
        return []
    merged: list[dict] = []
    start = float(episodes[0]["live_start"])
    end = float(episodes[0]["live_end"])
    raw_count = 1
    for episode in episodes[1:]:
        episode_start = float(episode["live_start"])
        episode_end = float(episode["live_end"])
        if episode_start - end <= maximum_gap_s + 1e-9:
            end = max(end, episode_end)
            raw_count += 1
            continue
        merged.append(_describe_bout(live_t, error_s, start, end, raw_count))
        start, end, raw_count = episode_start, episode_end, 1
    merged.append(_describe_bout(live_t, error_s, start, end, raw_count))
    return merged


def _describe_bout(
    live_t: np.ndarray,
    error_s: np.ndarray,
    start: float,
    end: float,
    raw_count: int,
) -> dict:
    inside = (live_t >= start) & (live_t <= end + 1e-9)
    indices = np.flatnonzero(inside)
    peak_index = indices[int(np.argmax(np.abs(error_s[inside])))]
    return {
        "live_start": start,
        "live_end": end,
        "elapsed_s": end - start,
        "constituent_episode_count": raw_count,
        "peak_live_t": float(live_t[peak_index]),
        "peak_error_s": float(error_s[peak_index]),
    }


def episode_duration_summary(episodes: list[dict]) -> dict:
    durations = np.asarray([item["duration_s"] for item in episodes], dtype=np.float64)
    if durations.size == 0:
        return {
            "count": 0,
            "median_s": None,
            "p90_s": None,
            "maximum_s": None,
            "fraction_at_most_2s": None,
            "fraction_at_most_5s": None,
        }
    return {
        "count": int(durations.size),
        "median_s": float(np.median(durations)),
        "p90_s": float(np.percentile(durations, 90)),
        "maximum_s": float(np.max(durations)),
        "fraction_at_most_2s": float(np.mean(durations <= 2.0 + 1e-9)),
        "fraction_at_most_5s": float(np.mean(durations <= 5.0 + 1e-9)),
    }


def local_tempo_predictors(sample_t: np.ndarray, ground_truth_ref_t: np.ndarray) -> dict:
    left = np.interp(sample_t - 1.0, sample_t, ground_truth_ref_t)
    right = np.interp(sample_t + 1.0, sample_t, ground_truth_ref_t)
    slope = (right - left) / 2.0
    log_slope = np.log(np.clip(slope, 1e-6, None))
    return {
        "absolute_log_local_slope": np.abs(log_slope),
        "absolute_log_slope_change": np.abs(np.gradient(log_slope)),
    }


def bias_sensitivity(live_t: np.ndarray, error_s: np.ndarray) -> dict:
    bias = float(np.median(error_s[np.isfinite(error_s)]))
    centred = error_s - bias
    original = failure_episodes(
        live_t,
        error_s,
        threshold_s=FAILURE_THRESHOLD_S,
        minimum_duration_s=MINIMUM_FAILURE_DURATION_S,
    )
    adjusted = failure_episodes(
        live_t,
        centred,
        threshold_s=FAILURE_THRESHOLD_S,
        minimum_duration_s=MINIMUM_FAILURE_DURATION_S,
    )
    original_duration = float(sum(item["duration_s"] for item in original))
    adjusted_duration = float(sum(item["duration_s"] for item in adjusted))
    return {
        "median_signed_bias_s": bias,
        "post_hoc_only": True,
        "original_episode_count": len(original),
        "original_failure_duration_s": original_duration,
        "centred_episode_count": len(adjusted),
        "centred_failure_duration_s": adjusted_duration,
        "failure_duration_reduction_fraction": (
            (original_duration - adjusted_duration) / original_duration
            if original_duration > 0
            else 0.0
        ),
    }


def acoustic_predictors(
    live_raw: np.ndarray,
    reference_raw: np.ndarray,
    *,
    trace: dict[str, np.ndarray],
    span: dict,
    center_to_availability_s: float,
) -> dict:
    live = normalize_features(live_raw)
    reference = normalize_features(reference_raw)
    absolute_end = min(float(span["live_end"]), float(trace["online_live_t"][-1]))
    sample_t = np.arange(math.ceil(float(span["live_start"])) + 1.0, absolute_end - 1.0, 1.0)
    relative_live_t = (
        sample_t
        - center_to_availability_s
        - float(span["live_start"])
    )
    absolute_error = np.abs(np.interp(sample_t, trace["online_live_t"], trace["error_s"]))
    ground_truth_ref_t = np.interp(
        sample_t,
        trace["online_live_t"],
        trace["ground_truth_ref_t"],
    )
    relative_ref_t = ground_truth_ref_t - float(span["ref_start"])
    live_index = np.clip(
        np.rint(relative_live_t * FRAME_RATE).astype(int),
        0,
        live.shape[0] - 1,
    )
    reference_index = np.clip(
        np.rint(relative_ref_t * FRAME_RATE).astype(int),
        0,
        reference.shape[0] - 1,
    )
    activity = np.linalg.norm(live_raw[live_index], axis=1)
    active_bins = np.sum(live_raw[live_index] > 0, axis=1).astype(np.float64)
    true_distance = np.linalg.norm(
        live[live_index] - reference[reference_index],
        axis=1,
    )
    alternative_distance = np.empty(sample_t.size, dtype=np.float64)
    candidate_count = np.empty(sample_t.size, dtype=np.float64)
    for index, (live_vector, centre) in enumerate(zip(live[live_index], reference_index)):
        low = max(0, centre - 10 * FRAME_RATE)
        high = min(reference.shape[0], centre + 10 * FRAME_RATE + 1)
        distances = np.linalg.norm(reference[low:high] - live_vector, axis=1)
        positions = np.arange(low, high)
        alternatives = distances[np.abs(positions - centre) > int(0.5 * FRAME_RATE)]
        alternative_distance[index] = (
            float(np.min(alternatives)) if alternatives.size else float("nan")
        )
        best = float(np.min(distances))
        candidate_count[index] = float(np.sum(distances <= best + 0.05))
    tempo = local_tempo_predictors(sample_t, ground_truth_ref_t)
    return {
        "sample_step_s": 1.0,
        "n_samples": int(sample_t.size),
        "correlation_uncertainty": {
            "method": "moving-block bootstrap",
            "block_s": BOOTSTRAP_BLOCK_S,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": RANDOM_SEED,
        },
        "predictors": {
            "cross_recording_true_path_distance": describe_predictor(
                true_distance,
                absolute_error,
            ),
            "local_alternative_advantage": describe_predictor(
                true_distance - alternative_distance,
                absolute_error,
            ),
            "near_best_candidate_count": describe_predictor(
                candidate_count,
                absolute_error,
            ),
            "unnormalized_onset_activity": describe_predictor(
                activity,
                absolute_error,
            ),
            "active_feature_bins": describe_predictor(
                active_bins,
                absolute_error,
            ),
            "absolute_log_local_slope": describe_predictor(
                tempo["absolute_log_local_slope"],
                absolute_error,
            ),
            "absolute_log_slope_change": describe_predictor(
                tempo["absolute_log_slope_change"],
                absolute_error,
            ),
        },
    }


def oracle_restart(
    live_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    live_start_s: float,
    reference_start_s: float,
    target_s: float,
    target_reference_s: float,
    span: dict,
    center_to_availability_s: float,
) -> dict:
    relative_live_start = (
        live_start_s
        - center_to_availability_s
        - float(span["live_start"])
    )
    relative_reference_start = reference_start_s - float(span["ref_start"])
    live_start_frame = int(round(relative_live_start * FRAME_RATE))
    reference_start_frame = int(round(relative_reference_start * FRAME_RATE))
    frames_to_target = int(round((target_s - live_start_s) * FRAME_RATE))
    target_reference_frame = int(
        round((target_reference_s - float(span["ref_start"])) * FRAME_RATE)
    )
    if live_start_frame < 0 or reference_start_frame < 0:
        raise ValueError("oracle restart starts outside the shared span")
    live_stop_frame = min(
        live_features.shape[0],
        live_start_frame + frames_to_target + 1,
    )
    reference_stop_frame = min(
        reference_features.shape[0],
        max(
            target_reference_frame + 30 * FRAME_RATE,
            reference_start_frame + 4 * frames_to_target + 1,
        ),
    )
    live_segment = live_features[live_start_frame:live_stop_frame]
    reference_segment = reference_features[reference_start_frame:reference_stop_frame]
    if live_segment.shape[0] <= frames_to_target or reference_segment.shape[0] < 2:
        raise ValueError("insufficient features for oracle restart")
    positions = run_dixon(reference_segment, live_segment)
    estimated_reference_s = (
        float(positions[frames_to_target])
        + float(span["ref_start"])
        + reference_start_frame / FRAME_RATE
    )
    signed_error = estimated_reference_s - target_reference_s
    coast_estimate = reference_start_s + (target_s - live_start_s)
    coast_error = coast_estimate - target_reference_s
    return {
        "restart_signed_error_s": signed_error,
        "restart_abs_error_s": abs(signed_error),
        "restart_success_within_1s": abs(signed_error) < FAILURE_THRESHOLD_S,
        "coast_signed_error_s": coast_error,
        "coast_abs_error_s": abs(coast_error),
        "coast_success_within_1s": abs(coast_error) < FAILURE_THRESHOLD_S,
    }


def run_oracle_resets(
    pair: str,
    bouts: list[dict],
    *,
    trace: dict[str, np.ndarray],
    live_features: np.ndarray,
    reference_features: np.ndarray,
    span: dict,
    center_to_availability_s: float,
) -> list[dict]:
    rows: list[dict] = []
    for bout_index, bout in enumerate(bouts, start=1):
        target_s = float(bout["peak_live_t"])
        target_reference_s = float(
            np.interp(target_s, trace["online_live_t"], trace["ground_truth_ref_t"])
        )
        for lead_s in RESET_LEADS_S:
            restart_s = target_s - lead_s
            row = {
                "pair": pair,
                "bout_index": bout_index,
                "bout_start_s": bout["live_start"],
                "bout_end_s": bout["live_end"],
                "constituent_episode_count": bout["constituent_episode_count"],
                "target_peak_live_s": target_s,
                "original_peak_signed_error_s": bout["peak_error_s"],
                "lead_s": lead_s,
                "available": restart_s >= (
                    float(span["live_start"]) + center_to_availability_s
                ),
            }
            if row["available"]:
                reference_restart_s = float(
                    np.interp(
                        restart_s,
                        trace["online_live_t"],
                        trace["ground_truth_ref_t"],
                    )
                )
                try:
                    row.update(
                        oracle_restart(
                            live_features,
                            reference_features,
                            live_start_s=restart_s,
                            reference_start_s=reference_restart_s,
                            target_s=target_s,
                            target_reference_s=target_reference_s,
                            span=span,
                            center_to_availability_s=center_to_availability_s,
                        )
                    )
                except ValueError as error:
                    row["available"] = False
                    row["unavailable_reason"] = str(error)
            rows.append(row)
    return rows


def summarize_oracle_rows(rows: list[dict]) -> dict:
    by_lead = {}
    for lead_s in RESET_LEADS_S:
        available = [
            row for row in rows if row["lead_s"] == lead_s and row["available"]
        ]
        if not available:
            by_lead[str(lead_s)] = {"n_bouts": 0}
            continue
        restart_errors = np.asarray(
            [row["restart_abs_error_s"] for row in available],
            dtype=np.float64,
        )
        coast_errors = np.asarray(
            [row["coast_abs_error_s"] for row in available],
            dtype=np.float64,
        )
        by_lead[str(lead_s)] = {
            "n_bouts": len(available),
            "restart_success_fraction_within_1s": float(np.mean(restart_errors < 1.0)),
            "coast_success_fraction_within_1s": float(np.mean(coast_errors < 1.0)),
            "restart_median_abs_error_s": float(np.median(restart_errors)),
            "coast_median_abs_error_s": float(np.median(coast_errors)),
            "restart_better_than_coast_fraction": float(
                np.mean(restart_errors < coast_errors)
            ),
        }
    return by_lead


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].astype(np.float64) for name in archive.files}


def circular_overlap_test(first: np.ndarray, second: np.ndarray) -> dict:
    observed = int(np.sum(first & second))
    null = np.asarray(
        [np.sum(first & np.roll(second, shift)) for shift in range(first.size)],
        dtype=np.float64,
    )
    null_mean = float(np.mean(null))
    observed_deviation = abs(observed - null_mean)
    p_value = float(
        (np.sum(np.abs(null - null_mean) >= observed_deviation - 1e-12) + 1)
        / (null.size + 1)
    )
    return {
        "observed_overlap_seconds": observed,
        "circular_null_mean_overlap_seconds": null_mean,
        "two_sided_exact_p": p_value,
    }


def cross_method_analysis() -> list[dict]:
    rows = []
    for pair, spec in PAIR_SPECS.items():
        traces = {}
        for method, suffix in METHOD_SUFFIXES.items():
            path = OUTPUT_ROOT / f"published_{spec['token']}_{suffix}_trace.npz"
            if path.exists():
                trace = load_trace(path)
                traces[method] = (trace["online_live_t"], trace["error_s"])
        methods = sorted(traces)
        for first_index, first_method in enumerate(methods):
            for second_method in methods[first_index + 1:]:
                first_t, first_error = traces[first_method]
                second_t, second_error = traces[second_method]
                start = math.ceil(max(float(first_t[0]), float(second_t[0])))
                end = math.floor(min(float(first_t[-1]), float(second_t[-1])))
                grid = np.arange(start, end + 1, dtype=np.float64)
                first_sample = np.interp(grid, first_t, first_error)
                second_sample = np.interp(grid, second_t, second_error)
                first_failed = np.abs(first_sample) >= FAILURE_THRESHOLD_S
                second_failed = np.abs(second_sample) >= FAILURE_THRESHOLD_S
                union = int(np.sum(first_failed | second_failed))
                row = {
                    "pair": pair,
                    "method_a": first_method,
                    "method_b": second_method,
                    "n_shared_seconds": int(grid.size),
                    "absolute_error_spearman_rho": spearman_rho(
                        np.abs(first_sample),
                        np.abs(second_sample),
                    ),
                    "failure_jaccard": (
                        float(np.sum(first_failed & second_failed) / union)
                        if union
                        else None
                    ),
                    "method_a_failure_seconds": int(np.sum(first_failed)),
                    "method_b_failure_seconds": int(np.sum(second_failed)),
                }
                row.update(circular_overlap_test(first_failed, second_failed))
                rows.append(row)
    return rows


def firebird_transition_analysis() -> list[dict]:
    overlay_path = OUTPUT_ROOT / "visual_transition_overlay_published_dixon.json"
    if not overlay_path.exists():
        return []
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    traces = {}
    for method, suffix in METHOD_SUFFIXES.items():
        path = OUTPUT_ROOT / f"published_firebird_{suffix}_trace.npz"
        if path.exists():
            traces[method] = load_trace(path)
    rows = []
    for transition in overlay["transitions"]:
        if transition.get("excluded"):
            continue
        live_t = float(transition["t"])
        row = {"segment": transition["seg"], "live_t": live_t, "methods": {}}
        for method, trace in traces.items():
            signed_tracker_error = float(
                np.interp(live_t, trace["online_live_t"], trace["error_s"])
            )
            transition_error = -signed_tracker_error
            row["methods"][method] = {
                "transition_error_s": transition_error,
                "fails_1s_threshold": abs(transition_error) >= FAILURE_THRESHOLD_S,
            }
        rows.append(row)
    return rows


def load_pair_inputs(pair: str, spec: dict) -> dict:
    result_path = OUTPUT_ROOT / f"published_{spec['token']}_matchmaker_dixon.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = REPOSITORY_ROOT / result["provenance"]["trace_path"]
    reference_path = OUTPUT_ROOT / spec["reference"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    preset = Preset.load(spec["preset"])
    for media_path in (preset.audio.live, preset.audio.ref):
        if not media_path.is_file():
            raise FileNotFoundError(f"source media is missing: {media_path}")
    return {
        "result_path": result_path,
        "result": result,
        "trace_path": trace_path,
        "trace": load_trace(trace_path),
        "reference_path": reference_path,
        "reference": reference,
        "preset": preset,
    }


def pair_analysis(pair: str, spec: dict) -> tuple[dict, list[dict]]:
    print(f"[{pair}] loading frozen trace and audited reference", flush=True)
    inputs = load_pair_inputs(pair, spec)
    result = inputs["result"]
    trace = inputs["trace"]
    reference = inputs["reference"]
    preset = inputs["preset"]
    span = reference["shared_span"]
    live_t = trace["online_live_t"]
    error_s = trace["error_s"]
    episodes = failure_episodes(
        live_t,
        error_s,
        threshold_s=FAILURE_THRESHOLD_S,
        minimum_duration_s=MINIMUM_FAILURE_DURATION_S,
    )
    bouts = merge_failure_bouts(live_t, error_s, episodes)
    peaks = np.asarray([item["peak_error_s"] for item in episodes], dtype=np.float64)
    failed_frames = np.abs(error_s) >= FAILURE_THRESHOLD_S

    print(f"[{pair}] extracting exact pinned Matchmaker features", flush=True)
    reference_raw, reference_feature_runtime = compute_exact_features(
        preset.audio.ref,
        float(span["ref_start"]),
        float(span["ref_end"]),
    )
    live_raw, live_feature_runtime = compute_exact_features(
        preset.audio.live,
        float(span["live_start"]),
        float(span["live_end"]),
    )
    reference_features = normalize_features(reference_raw)
    live_features = normalize_features(live_raw)

    calibration = dict(result["timing_calibration"])
    calibration_bias = float(
        calibration["raw_position_relative_to_availability"][
            "median_signed_error_s"
        ]
    )
    calibration_adjusted = failure_episodes(
        live_t,
        error_s - calibration_bias,
        threshold_s=FAILURE_THRESHOLD_S,
        minimum_duration_s=MINIMUM_FAILURE_DURATION_S,
    )
    calibration["cross_recording_sensitivity"] = {
        "original_episode_count": len(episodes),
        "calibration_adjusted_episode_count": len(calibration_adjusted),
        "original_failure_duration_s": float(
            sum(item["duration_s"] for item in episodes)
        ),
        "calibration_adjusted_failure_duration_s": float(
            sum(item["duration_s"] for item in calibration_adjusted)
        ),
    }

    print(f"[{pair}] testing acoustic and tempo predictors", flush=True)
    predictor_results = acoustic_predictors(
        live_raw,
        reference_raw,
        trace=trace,
        span=span,
        center_to_availability_s=float(
            result["tracker"]["feature_timing"]["center_to_availability_s"]
        ),
    )
    del live_raw, reference_raw

    print(f"[{pair}] restarting every merged failure bout", flush=True)
    oracle_rows = run_oracle_resets(
        pair,
        bouts,
        trace=trace,
        live_features=live_features,
        reference_features=reference_features,
        span=span,
        center_to_availability_s=float(
            result["tracker"]["feature_timing"]["center_to_availability_s"]
        ),
    )
    del live_features, reference_features

    pair_payload = {
        "pair": pair,
        "inputs": {
            "result": {
                "path": relative_path(inputs["result_path"]),
                "sha256": sha256(inputs["result_path"]),
            },
            "trace": {
                "path": relative_path(inputs["trace_path"]),
                "sha256": sha256(inputs["trace_path"]),
            },
            "audited_reference": {
                "path": relative_path(inputs["reference_path"]),
                "sha256": sha256(inputs["reference_path"]),
            },
            "live_media": {
                "filename": preset.audio.live.name,
                "size_bytes": preset.audio.live.stat().st_size,
                "sha256": sha256(preset.audio.live),
            },
            "reference_media": {
                "filename": preset.audio.ref.name,
                "size_bytes": preset.audio.ref.stat().st_size,
                "sha256": sha256(preset.audio.ref),
            },
            "shared_span": span,
        },
        "published_baseline_metrics": result["metrics"],
        "failure_direction": {
            "median_signed_error_s": float(np.median(error_s)),
            "failed_frame_count": int(np.sum(failed_frames)),
            "behind_fraction_of_failed_frames": (
                float(np.mean(error_s[failed_frames] < 0))
                if np.any(failed_frames)
                else None
            ),
            "episode_peak_behind_count": int(np.sum(peaks < 0)),
            "episode_peak_ahead_count": int(np.sum(peaks > 0)),
        },
        "episode_duration": episode_duration_summary(episodes),
        "merged_bouts": {
            "maximum_gap_s": MERGE_GAP_S,
            "count": len(bouts),
            "constituent_episode_count": len(episodes),
        },
        "post_hoc_fixed_bias_sensitivity": bias_sensitivity(live_t, error_s),
        "latency_calibration": calibration,
        "acoustic_and_tempo_predictors": predictor_results,
        "oracle_reset_summary": summarize_oracle_rows(oracle_rows),
        "runtime": {
            "reference_feature_extraction_s": reference_feature_runtime,
            "live_feature_extraction_s": live_feature_runtime,
        },
    }
    return pair_payload, oracle_rows


def aggregate_pairs(pair_results: list[dict], oracle_rows: list[dict]) -> dict:
    behind = sum(
        item["failure_direction"]["episode_peak_behind_count"] for item in pair_results
    )
    ahead = sum(
        item["failure_direction"]["episode_peak_ahead_count"] for item in pair_results
    )
    durations = []
    for spec in PAIR_SPECS.values():
        result_path = OUTPUT_ROOT / f"published_{spec['token']}_matchmaker_dixon.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        durations.extend(float(item["duration_s"]) for item in result["episodes"])
    duration_array = np.asarray(durations, dtype=np.float64)
    original_failure_duration = sum(
        item["post_hoc_fixed_bias_sensitivity"]["original_failure_duration_s"]
        for item in pair_results
    )
    centred_failure_duration = sum(
        item["post_hoc_fixed_bias_sensitivity"]["centred_failure_duration_s"]
        for item in pair_results
    )
    return {
        "failure_episode_direction": {
            "behind": behind,
            "ahead": ahead,
            "total": behind + ahead,
            "behind_fraction": behind / (behind + ahead) if behind + ahead else None,
        },
        "failure_episode_duration": {
            "count": int(duration_array.size),
            "median_s": float(np.median(duration_array)),
            "p90_s": float(np.percentile(duration_array, 90)),
            "maximum_s": float(np.max(duration_array)),
            "fraction_at_most_2s": float(np.mean(duration_array <= 2.0 + 1e-9)),
            "fraction_at_most_5s": float(np.mean(duration_array <= 5.0 + 1e-9)),
        },
        "merged_bouts": {
            "count": sum(item["merged_bouts"]["count"] for item in pair_results),
            "constituent_episode_count": sum(
                item["merged_bouts"]["constituent_episode_count"]
                for item in pair_results
            ),
            "maximum_gap_s": MERGE_GAP_S,
        },
        "post_hoc_fixed_bias_sensitivity": {
            "original_failure_duration_s": original_failure_duration,
            "centred_failure_duration_s": centred_failure_duration,
            "failure_duration_reduction_fraction": (
                (original_failure_duration - centred_failure_duration)
                / original_failure_duration
                if original_failure_duration
                else 0.0
            ),
        },
        "oracle_reset_summary": summarize_oracle_rows(oracle_rows),
    }


def write_oracle_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "pair",
        "bout_index",
        "bout_start_s",
        "bout_end_s",
        "constituent_episode_count",
        "target_peak_live_s",
        "original_peak_signed_error_s",
        "lead_s",
        "available",
        "restart_signed_error_s",
        "restart_abs_error_s",
        "restart_success_within_1s",
        "coast_signed_error_s",
        "coast_abs_error_s",
        "coast_success_within_1s",
        "unavailable_reason",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def format_percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_report(payload: dict) -> str:
    aggregate = payload["aggregate"]
    direction = aggregate["failure_episode_direction"]
    duration = aggregate["failure_episode_duration"]
    lines = [
        "# Audio-failure analysis",
        "",
        "This directory is generated by `shared/analysis/audio_failure_analysis.py`. It does not modify or build the thesis.",
        "",
        "## Protocol",
        "",
        f"- Failure: absolute error at least {FAILURE_THRESHOLD_S:.1f} s for at least {MINIMUM_FAILURE_DURATION_S:.1f} s.",
        f"- Bout: failure episodes separated by at most {MERGE_GAP_S:.1f} s are merged.",
        "- Oracle-reset target: the worst-error frame in each merged bout.",
        "- Reset leads: 2, 5, 10, and 20 seconds before the target.",
        "- Control: 1:1 progression from the same oracle position without audio tracking.",
        "- Predictor uncertainty: 30-second moving-block bootstrap; recordings remain the inferential unit.",
        "",
        "## Reproduced descriptive results",
        "",
        f"- Episode peaks behind: {direction['behind']}/{direction['total']} ({format_percentage(direction['behind_fraction'])}).",
        f"- Episode duration: median {duration['median_s']:.2f} s; 90th percentile {duration['p90_s']:.2f} s; {format_percentage(duration['fraction_at_most_5s'])} at most 5 s.",
        f"- Raw episodes merge from {aggregate['merged_bouts']['constituent_episode_count']} to {aggregate['merged_bouts']['count']} bouts.",
        "",
        "## Latency calibration",
        "",
        f"The centered {FFT_SAMPLES}-sample feature window requires {HOP_SAMPLES / SAMPLE_RATE * 1000:.1f} ms of future audio before a frame centered at the current time is available. This is acquisition delay, not a position bias.",
        "",
    ]
    for pair in payload["pairs"]:
        calibration = pair["latency_calibration"]
        center = calibration["raw_position_relative_to_feature_center"]
        availability = calibration["raw_position_relative_to_availability"]
        lines.append(
            f"- {pair['pair']}: identical-input median bias is "
            f"{center['median_signed_error_s']:+.4f} s at feature-centre time "
            f"and {availability['median_signed_error_s']:+.4f} s at causal "
            "availability time."
        )
    lines.extend(["", "## Oracle resets", ""])
    for lead, result in aggregate["oracle_reset_summary"].items():
        if result["n_bouts"] == 0:
            continue
        lines.append(
            f"- {lead} s lead ({result['n_bouts']} bouts): tracker reset succeeds "
            f"{format_percentage(result['restart_success_fraction_within_1s'])}; "
            f"coasting succeeds {format_percentage(result['coast_success_fraction_within_1s'])}; "
            f"tracker beats coasting {format_percentage(result['restart_better_than_coast_fraction'])}."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Oracle resets are diagnostic interventions, not a deployable system: they receive the audited correct position.",
            "- A successful short-lead reset establishes trajectory dependence but does not by itself prove that local audio can recover global position.",
            "- Same-input calibration measures decoder/indexing bias. It does not measure audio-driver, buffering, operating-system, or output-device latency.",
            "- Scalar predictor results are descriptive across four recording pairs and do not establish causal musical conditions.",
            "- Post-hoc median-bias subtraction uses the complete run and is not a causal online correction.",
            "",
            "See `summary.json` for exact input hashes and full results, and `oracle_resets.csv` for every intervention.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = {
        "summary": output_dir / "summary.json",
        "oracle": output_dir / "oracle_resets.csv",
        "report": output_dir / "README.md",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to replace generated output without --force: {joined}")
    version = importlib.metadata.version("pymatchmaker")
    if version != MATCHMAKER_VERSION:
        raise SystemExit(
            f"expected pymatchmaker {MATCHMAKER_VERSION}, found {version}; "
            "use .cache-published-baselines/matchmaker-venv/bin/python"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pair_results = []
    oracle_rows = []
    for pair, spec in PAIR_SPECS.items():
        pair_result, pair_oracle_rows = pair_analysis(pair, spec)
        pair_results.append(pair_result)
        oracle_rows.extend(pair_oracle_rows)
    payload = {
        "schema_version": 1,
        "status": "exploratory_reproducible_analysis_not_yet_thesis_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": relative_path(Path(__file__)),
            "script_sha256": sha256(Path(__file__)),
            "repository_revision": repository_revision(),
            "pymatchmaker_version": version,
            "pymatchmaker_release_commit": MATCHMAKER_COMMIT,
            "numpy_version": importlib.metadata.version("numpy"),
            "scipy_version": importlib.metadata.version("scipy"),
            "random_seed": RANDOM_SEED,
            "total_runtime_s": time.perf_counter() - started,
        },
        "protocol": {
            "failure_absolute_error_threshold_s": FAILURE_THRESHOLD_S,
            "minimum_failure_duration_s": MINIMUM_FAILURE_DURATION_S,
            "merge_gap_s": MERGE_GAP_S,
            "oracle_reset_leads_s": RESET_LEADS_S,
            "oracle_target": "maximum absolute original error in each merged bout",
            "oracle_control": "1:1 reference-time progression from the same audited start position",
            "feature": {
                "name": "Matchmaker LogSpectralEnergyProcessor",
                "sample_rate_hz": SAMPLE_RATE,
                "hop_samples": HOP_SAMPLES,
                "fft_samples": FFT_SAMPLES,
                "frame_rate_hz": FRAME_RATE,
                "centered_feature_acquisition_delay_s": HOP_SAMPLES / SAMPLE_RATE,
            },
            "uncertainty": {
                "method": "moving-block bootstrap",
                "block_s": BOOTSTRAP_BLOCK_S,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": RANDOM_SEED,
                "naive_per_second_p_values": False,
            },
        },
        "aggregate": aggregate_pairs(pair_results, oracle_rows),
        "pairs": pair_results,
        "cross_method_failure_overlap": cross_method_analysis(),
        "firebird_transition_methods": firebird_transition_analysis(),
        "interpretation_limits": [
            "Oracle resets receive audited ground-truth state and are diagnostic rather than deployable.",
            "Successful short-lead resets show trajectory dependence but do not prove local audio can globally relocalize.",
            "Identical-input calibration measures decoder/indexing bias, not hardware or operating-system latency.",
            "Post-hoc bias correction is non-causal because it uses the complete run.",
            "Four recording pairs support descriptive replication, not causal attribution to musical properties.",
        ],
    }
    write_json_atomic(outputs["summary"], payload)
    write_oracle_csv(outputs["oracle"], oracle_rows)
    write_text_atomic(outputs["report"], render_report(payload))
    print(f"wrote {outputs['summary']}", flush=True)
    print(f"wrote {outputs['oracle']}", flush=True)
    print(f"wrote {outputs['report']}", flush=True)


if __name__ == "__main__":
    main()
