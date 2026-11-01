#!/usr/bin/env python3
"""Compare published Dixon audio tracking with author-approved oracle cue resets.

This script implements only the fixed visual-contribution paired-cue experiment. It does not
run the automatic cue detector or the visual-tempo experiment.
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
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = _AUDIO_ROOT
REPOSITORY_ROOT = _REPO_ROOT
SCRIPTS = _REPO_ROOT / "shared/analysis"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from analyze_audio_failures import error_metrics, repository_revision, sha256  # noqa: E402
from run_published_tracker import (  # noqa: E402
    MATCHMAKER_COMMIT,
    MATCHMAKER_VERSION,
    compute_matchmaker_features_chunked,
    decode_matchmaker,
    load_span_direct,
    make_matchmaker_follower,
    write_json,
)


FRAME_RATE = 30
SAMPLE_RATE = 44_100
HOP_SAMPLES = 1_470
METHOD = "external_matchmaker_dixon_v0_3_0"
VIDEO_FRAME_S = 1 / 25
PRIMARY_EXCLUSIONS = {"139", "140"}
ACCEPTABLE_EARLY_S = -0.60
ACCEPTABLE_LATE_S = 0.33
LIMITS_S = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Transition:
    segment: str
    live_target_s: float
    reference_target_s: float


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def external_media_record(path: Path) -> dict:
    """Identify non-repository media without publishing a local machine path."""

    return {
        "filename": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "output/firebird_reference.json",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=REPOSITORY_ROOT / "rq2-conductor-cues/detector/transition_cue_pairing.v1.json",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=REPOSITORY_ROOT / "rq2-conductor-cues/features/firebird_timeline.txt",
    )
    parser.add_argument(
        "--approved-baseline-trace",
        type=Path,
        default=PROJECT_ROOT / "output/published_firebird_matchmaker_dixon_trace.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output/visual_paired_cue_resets_v1",
    )
    return parser.parse_args()


def load_dense_reference(reference_path: Path) -> tuple[dict, Path, np.ndarray, np.ndarray]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not reference.get("structural_qc", {}).get("passed", False):
        raise RuntimeError("dense reference did not pass structural QC")
    dense_path = reference_path.parent / reference["dense_path"]
    with np.load(dense_path) as archive:
        live_t = archive["live_t"].astype(np.float64)
        ref_t = archive["ref_t"].astype(np.float64)
    return reference, dense_path, live_t, ref_t


def parse_timeline(
    path: Path,
    dense_live_t: np.ndarray,
    dense_ref_t: np.ndarray,
) -> list[Transition]:
    transitions: list[Transition] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("@", "")
        if not line or line.startswith("#"):
            continue
        segment, timestamp = line.split()[:2]
        minutes, seconds, frames = map(int, timestamp.split(":"))
        live_target_s = minutes * 60 + seconds + frames / 25
        reference_target_s = float(
            np.interp(live_target_s, dense_live_t, dense_ref_t)
        )
        transitions.append(Transition(segment, live_target_s, reference_target_s))
    if not transitions:
        raise ValueError("timeline is empty")
    return transitions


def first_reach_time(
    live_t: np.ndarray,
    reference_position_t: np.ndarray,
    target_reference_s: float,
) -> float | None:
    """Return the first causal output time at or beyond a reference target."""

    reached = np.flatnonzero(reference_position_t >= target_reference_s)
    return float(live_t[reached[0]]) if reached.size else None


def reference_frame_at_or_before(
    reference_s: float,
    *,
    span_start_s: float,
    frame_rate: int,
    frame_count: int,
) -> int:
    """Quantize a reset without placing the tracker ahead of its oracle state."""

    relative = (reference_s - span_start_s) * frame_rate
    return max(0, min(frame_count - 1, int(np.floor(relative + 1e-12))))


def transition_rows(
    transitions: list[Transition],
    live_t: np.ndarray,
    audio_ref_t: np.ndarray,
    reset_ref_t: np.ndarray,
    paired_segments: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for transition in transitions:
        audio_fire = first_reach_time(
            live_t, audio_ref_t, transition.reference_target_s
        )
        reset_fire = first_reach_time(
            live_t, reset_ref_t, transition.reference_target_s
        )
        audio_error = (
            audio_fire - transition.live_target_s if audio_fire is not None else None
        )
        reset_error = (
            reset_fire - transition.live_target_s if reset_fire is not None else None
        )
        row = {
            "segment": transition.segment,
            "included_primary": transition.segment not in PRIMARY_EXCLUSIONS,
            "has_approved_pair": transition.segment in paired_segments,
            "marked_live_target_s": transition.live_target_s,
            "reference_target_s": transition.reference_target_s,
            "audio_only": describe_transition(audio_fire, audio_error),
            "perfect_paired_cue_reset": describe_transition(reset_fire, reset_error),
        }
        row["effect"] = classify_effect(audio_error, reset_error)
        rows.append(row)
    return rows


def is_acceptable(error_s: float | None) -> bool:
    return (
        error_s is not None
        and ACCEPTABLE_EARLY_S <= error_s <= ACCEPTABLE_LATE_S
    )


def describe_transition(fire_s: float | None, error_s: float | None) -> dict:
    return {
        "fire_s": fire_s,
        "signed_error_s": error_s,
        "absolute_error_s": abs(error_s) if error_s is not None else None,
        "acceptable": is_acceptable(error_s),
        "within_limits": {
            str(limit): bool(error_s is not None and abs(error_s) <= limit)
            for limit in LIMITS_S
        },
    }


def classify_effect(audio_error: float | None, reset_error: float | None) -> str:
    if audio_error is None or reset_error is None:
        return "unreached_target"
    audio_ok = is_acceptable(audio_error)
    reset_ok = is_acceptable(reset_error)
    if not audio_ok and reset_ok:
        return "acceptable_gain"
    if audio_ok and not reset_ok:
        return "acceptable_harm"
    change = abs(audio_error) - abs(reset_error)
    if change > VIDEO_FRAME_S + 1e-9:
        return "improved_same_acceptability"
    if change < -VIDEO_FRAME_S - 1e-9:
        return "worsened_same_acceptability"
    return "no_material_change"


def summarize_transition_rows(rows: list[dict]) -> dict:
    def summarize(selected: list[dict]) -> dict:
        output: dict = {"n_targets": len(selected)}
        for condition in ("audio_only", "perfect_paired_cue_reset"):
            errors = np.asarray(
                [row[condition]["signed_error_s"] for row in selected],
                dtype=np.float64,
            )
            finite = np.isfinite(errors)
            usable = errors[finite]
            output[condition] = {
                "n_reached": int(usable.size),
                "median_absolute_error_s": (
                    float(np.median(np.abs(usable))) if usable.size else None
                ),
                "mean_absolute_error_s": (
                    float(np.mean(np.abs(usable))) if usable.size else None
                ),
                "acceptable_count": int(
                    sum(row[condition]["acceptable"] for row in selected)
                ),
                "within_limit_count": {
                    str(limit): int(
                        sum(row[condition]["within_limits"][str(limit)] for row in selected)
                    )
                    for limit in LIMITS_S
                },
            }
        effect_names = sorted({row["effect"] for row in selected})
        output["effects"] = {
            name: int(sum(row["effect"] == name for row in selected))
            for name in effect_names
        }
        return output

    return {
        "primary_14": summarize([row for row in rows if row["included_primary"]]),
        "sensitivity_all_16": summarize(rows),
        "approved_pairs_only_4": summarize(
            [row for row in rows if row["has_approved_pair"]]
        ),
    }


def decode_with_oracle_resets(
    reference_features: np.ndarray,
    live_features: np.ndarray,
    availability_live_t: np.ndarray,
    pairs: list[dict],
    dense_live_t: np.ndarray,
    dense_ref_t: np.ndarray,
    *,
    reference_span_start_s: float,
) -> tuple[np.ndarray, list[dict]]:
    """Run Dixon causally, replacing its state at each approved cue."""

    reset_specs = []
    previous_index = -1
    for pair in sorted(pairs, key=lambda item: float(item["cue_time_s"])):
        cue_s = float(pair["cue_time_s"])
        reset_index = int(np.searchsorted(availability_live_t, cue_s, side="left"))
        if reset_index >= live_features.shape[0]:
            raise ValueError(f"cue {pair['cue_id']} is outside the live feature stream")
        if reset_index <= previous_index:
            raise ValueError("approved cues do not map to unique increasing reset frames")
        requested_ref_s = float(np.interp(cue_s, dense_live_t, dense_ref_t))
        relative_frame = reference_frame_at_or_before(
            requested_ref_s,
            span_start_s=reference_span_start_s,
            frame_rate=FRAME_RATE,
            frame_count=reference_features.shape[0],
        )
        reset_specs.append(
            {
                "segment": str(pair["segment"]),
                "cue_id": pair["cue_id"],
                "cue_label": pair["cue_label"],
                "cue_timestamp_s": cue_s,
                "reset_frame_index": reset_index,
                "reset_available_s": float(availability_live_t[reset_index]),
                "reset_delay_from_cue_s": float(availability_live_t[reset_index] - cue_s),
                "requested_reference_s": requested_ref_s,
                "quantized_reference_s": (
                    reference_span_start_s + relative_frame / FRAME_RATE
                ),
                "reference_slice_start_frame": relative_frame,
            }
        )
        previous_index = reset_index

    positions = np.empty(live_features.shape[0], dtype=np.float64)
    boundaries = [0] + [item["reset_frame_index"] for item in reset_specs]
    stops = [item["reset_frame_index"] for item in reset_specs] + [live_features.shape[0]]
    offsets = [0] + [item["reference_slice_start_frame"] for item in reset_specs]
    for start, stop, offset in zip(boundaries, stops, offsets):
        if stop <= start:
            continue
        follower, _, _ = make_matchmaker_follower(
            reference_features[offset:],
            method=METHOD,
            frame_rate=FRAME_RATE,
        )
        last_position = float(offset) / FRAME_RATE
        for index in range(start, stop):
            if follower.is_still_following():
                relative_position = float(
                    follower(live_features[index], index / FRAME_RATE)
                )
                last_position = offset / FRAME_RATE + relative_position
            positions[index] = reference_span_start_s + last_position
        if hasattr(follower, "input_features"):
            follower.input_features.clear()
        if hasattr(follower, "_alignment_path"):
            follower._alignment_path.clear()
    return positions, reset_specs


def dense_metrics(error_s: np.ndarray) -> dict:
    metrics = error_metrics(error_s, threshold=1.0)
    metrics["within_limits"] = {
        str(limit): float(np.mean(np.abs(error_s) <= limit)) for limit in LIMITS_S
    }
    return metrics


def path_diagnostics(reference_position_s: np.ndarray) -> dict:
    steps = np.diff(reference_position_s)
    backwards = steps < -1e-12
    return {
        "backward_step_count": int(np.sum(backwards)),
        "maximum_backward_step_s": (
            float(np.max(-steps[backwards])) if np.any(backwards) else 0.0
        ),
        "note": (
            "Published Dixon positions are not constrained to be monotonic. "
            "The one-shot transition controller does not undo a target after firing."
        ),
    }


def cue_window_rows(
    reset_specs: list[dict],
    live_t: np.ndarray,
    audio_error_s: np.ndarray,
    reset_error_s: np.ndarray,
) -> list[dict]:
    rows = []
    windows = (("before_2s", -2.0, 0.0), ("after_2s", 0.0, 2.0), ("after_2_to_5s", 2.0, 5.0))
    for spec in reset_specs:
        cue_s = spec["cue_timestamp_s"]
        summaries = {}
        for name, left, right in windows:
            inside = (live_t >= cue_s + left) & (live_t < cue_s + right)
            summaries[name] = {}
            for condition, error in (
                ("audio_only", audio_error_s),
                ("perfect_paired_cue_reset", reset_error_s),
            ):
                values = np.abs(error[inside])
                summaries[name][condition] = {
                    "n_frames": int(values.size),
                    "median_absolute_error_s": (
                        float(np.median(values)) if values.size else None
                    ),
                    "mean_absolute_error_s": (
                        float(np.mean(values)) if values.size else None
                    ),
                }
        rows.append({**spec, "dense_error_windows": summaries})
    return rows


def baseline_agreement(
    live_t: np.ndarray,
    recomputed_ref_t: np.ndarray,
    approved_trace_path: Path,
) -> dict:
    with np.load(approved_trace_path) as archive:
        approved_live_t = archive["online_live_t"].astype(np.float64)
        approved_ref_t = archive["online_ref_t"].astype(np.float64)
    maximum_time_difference_s = (
        float(np.max(np.abs(live_t - approved_live_t)))
        if live_t.shape == approved_live_t.shape
        else None
    )
    same_grid = bool(
        live_t.shape == approved_live_t.shape
        and np.allclose(live_t, approved_live_t, rtol=0.0, atol=1e-12)
    )
    if same_grid:
        difference = recomputed_ref_t - approved_ref_t
    else:
        difference = recomputed_ref_t - np.interp(live_t, approved_live_t, approved_ref_t)
    return {
        "same_time_grid": same_grid,
        "max_absolute_timestamp_difference_s": maximum_time_difference_s,
        "max_absolute_reference_position_difference_s": float(np.max(np.abs(difference))),
        "median_absolute_reference_position_difference_s": float(np.median(np.abs(difference))),
    }


def main() -> None:
    args = parse_args()
    for path in (args.live, args.ref, args.reference, args.pairs, args.timeline):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    trace_path = args.output_dir / "causal_paths.npz"
    if result_path.exists() or trace_path.exists():
        raise SystemExit(
            f"refusing to overwrite versioned visual-contribution output directory: {args.output_dir}"
        )

    reference, dense_path, dense_live_t, dense_ref_t = load_dense_reference(
        args.reference
    )
    span = reference["shared_span"]
    pairs_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    if pairs_payload.get("status") != "approved_primary_pair_whitelist":
        raise RuntimeError("pairing file is not the approved primary whitelist")
    pairs = pairs_payload["primary_pair_whitelist"]
    transitions = parse_timeline(args.timeline, dense_live_t, dense_ref_t)

    from matchmaker.features.audio import LogSpectralEnergyProcessor

    started = time.perf_counter()
    processor = LogSpectralEnergyProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_SAMPLES,
    )
    reference_audio = load_span_direct(
        args.ref,
        start=float(span["ref_start"]),
        end=float(span["ref_end"]),
        sr=SAMPLE_RATE,
    )
    reference_features = compute_matchmaker_features_chunked(
        reference_audio, processor=processor, hop=HOP_SAMPLES
    )
    del reference_audio
    live_audio = load_span_direct(
        args.live,
        start=float(span["live_start"]),
        end=float(span["live_end"]),
        sr=SAMPLE_RATE,
    )
    live_features = compute_matchmaker_features_chunked(
        live_audio, processor=processor, hop=HOP_SAMPLES
    )
    del live_audio
    feature_runtime_s = time.perf_counter() - started

    availability_live_t = (
        np.arange(live_features.shape[0], dtype=np.float64) / FRAME_RATE
        + HOP_SAMPLES / SAMPLE_RATE
        + float(span["live_start"])
    )
    decode_started = time.perf_counter()
    baseline_relative, _, _, _ = decode_matchmaker(
        reference_features,
        live_features,
        method=METHOD,
        frame_rate=FRAME_RATE,
        progress=False,
    )
    baseline_ref_t = baseline_relative + float(span["ref_start"])
    reset_ref_t, reset_specs = decode_with_oracle_resets(
        reference_features,
        live_features,
        availability_live_t,
        pairs,
        dense_live_t,
        dense_ref_t,
        reference_span_start_s=float(span["ref_start"]),
    )
    decode_runtime_s = time.perf_counter() - decode_started

    inside = (
        (availability_live_t >= dense_live_t[0])
        & (availability_live_t <= dense_live_t[-1])
    )
    live_t = availability_live_t[inside]
    baseline_ref_t = baseline_ref_t[inside]
    reset_ref_t = reset_ref_t[inside]
    ground_truth_ref_t = np.interp(live_t, dense_live_t, dense_ref_t)
    audio_error_s = baseline_ref_t - ground_truth_ref_t
    reset_error_s = reset_ref_t - ground_truth_ref_t

    paired_segments = {str(pair["segment"]) for pair in pairs}
    rows = transition_rows(
        transitions, live_t, baseline_ref_t, reset_ref_t, paired_segments
    )
    reset_specs = cue_window_rows(
        reset_specs, live_t, audio_error_s, reset_error_s
    )
    for spec in reset_specs:
        pair = next(item for item in pairs if item["cue_id"] == spec["cue_id"])
        spec["cue_minus_transition_target_s"] = (
            float(pair["cue_time_s"]) - float(pair["target_time_s"])
        )

    agreement = baseline_agreement(
        live_t, baseline_ref_t, args.approved_baseline_trace
    )
    if not agreement["same_time_grid"] or agreement[
        "max_absolute_reference_position_difference_s"
    ] > 1e-5:
        raise RuntimeError(
            "recomputed audio-only baseline does not reproduce the approved trace: "
            f"{agreement}"
        )

    payload = {
        "schema_version": 1,
        "status": "visual_contribution_paired_cue_oracle_reset_complete",
        "scope": "Firebird production-specific case study; four author-approved cue-transition pairs",
        "claim_limit": (
            "Oracle ceiling only: perfect cue occurrence and correct reference landmark "
            "are supplied from approved annotations and dense offline alignment."
        ),
        "method": {
            "audio_tracker": METHOD,
            "implementation": "pymatchmaker",
            "version": MATCHMAKER_VERSION,
            "release_commit": MATCHMAKER_COMMIT,
            "feature": "upstream log-spectral energy",
            "sample_rate_hz": SAMPLE_RATE,
            "hop_samples": HOP_SAMPLES,
            "frame_rate_hz": FRAME_RATE,
            "evaluation_timestamp": "complete audio-window availability",
            "reset_rule": (
                "At the first 30 Hz audio output available at or after each approved cue, "
                "restart the unchanged published Dixon follower on a reference slice whose "
                "first frame is the dense-alignment reference position at the cue."
            ),
            "reset_reference_quantization": (
                "30 Hz reference feature frame at or before the oracle position; "
                "this cannot seed the tracker ahead of the known state and matches "
                "the existing live snap_to floor operation"
            ),
            "transition_fire_rule": (
                "first causal tracker output at or beyond each transition reference target"
            ),
            "transition_controller": (
                "ordered one-shot targets; a fired transition is not undone if the "
                "published tracker later reports a lower reference position"
            ),
            "transition_error": "fire time minus marked live target time",
            "acceptable_band_s": [ACCEPTABLE_EARLY_S, ACCEPTABLE_LATE_S],
            "absolute_error_limits_s": list(LIMITS_S),
            "material_change_s": VIDEO_FRAME_S,
            "primary_excluded_segments": sorted(PRIMARY_EXCLUSIONS),
        },
        "baseline_reproduction_check": agreement,
        "dense_path_metrics": {
            "audio_only": dense_metrics(audio_error_s),
            "perfect_paired_cue_reset": dense_metrics(reset_error_s),
        },
        "path_diagnostics": {
            "audio_only": path_diagnostics(baseline_ref_t),
            "perfect_paired_cue_reset": path_diagnostics(reset_ref_t),
        },
        "approved_resets": reset_specs,
        "transition_summary": summarize_transition_rows(rows),
        "transitions": rows,
        "provenance": {
            "repository_revision_before_output": repository_revision(),
            "feature_runtime_s": feature_runtime_s,
            "decode_runtime_s": decode_runtime_s,
            "inputs": {
                "live_audio": external_media_record(args.live),
                "reference_audio": external_media_record(args.ref),
                "reference": {"path": repository_path(args.reference), "sha256": sha256(args.reference)},
                "dense_reference": {"path": repository_path(dense_path), "sha256": sha256(dense_path)},
                "approved_pairs": {"path": repository_path(args.pairs), "sha256": sha256(args.pairs)},
                "timeline": {"path": repository_path(args.timeline), "sha256": sha256(args.timeline)},
                "approved_baseline_trace": {"path": repository_path(args.approved_baseline_trace), "sha256": sha256(args.approved_baseline_trace)},
            },
            "trace_path": repository_path(trace_path),
        },
    }

    write_json(result_path, payload)
    temporary_trace = trace_path.with_suffix(trace_path.suffix + ".tmp")
    with temporary_trace.open("wb") as handle:
        np.savez_compressed(
            handle,
            online_live_t=live_t,
            ground_truth_ref_t=ground_truth_ref_t,
            audio_only_ref_t=baseline_ref_t,
            audio_only_error_s=audio_error_s,
            perfect_paired_cue_reset_ref_t=reset_ref_t,
            perfect_paired_cue_reset_error_s=reset_error_s,
        )
    temporary_trace.replace(trace_path)
    print(json.dumps(payload["transition_summary"], indent=2))
    print(f"wrote {result_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
