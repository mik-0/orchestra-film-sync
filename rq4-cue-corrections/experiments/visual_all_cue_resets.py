#!/usr/bin/env python3
"""Run the visual-contribution all-cue oracle and automatic landmark-reset experiment."""

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
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = _AUDIO_ROOT
REPOSITORY_ROOT = _REPO_ROOT
SCRIPTS = _REPO_ROOT / "shared/analysis"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from analyze_audio_failures import repository_revision, sha256  # noqa: E402
from visual_paired_cue_resets import (  # noqa: E402
    ACCEPTABLE_EARLY_S,
    ACCEPTABLE_LATE_S,
    FRAME_RATE,
    HOP_SAMPLES,
    LIMITS_S,
    METHOD,
    PRIMARY_EXCLUSIONS,
    SAMPLE_RATE,
    Transition,
    baseline_agreement,
    classify_effect,
    dense_metrics,
    describe_transition,
    external_media_record,
    load_dense_reference,
    parse_timeline,
    path_diagnostics,
    reference_frame_at_or_before,
    repository_path,
)
from run_published_tracker import (  # noqa: E402
    MATCHMAKER_COMMIT,
    MATCHMAKER_VERSION,
    compute_matchmaker_features_chunked,
    decode_matchmaker,
    load_span_direct,
    make_matchmaker_follower,
)


ELIGIBILITY_WIDTHS_S = (0.5, 1.0, 2.0, 4.0)
STABILITY_HOLDS_S = (0.0, 0.12, 0.20, 0.50)


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
        "--annotations",
        type=Path,
        default=REPOSITORY_ROOT / "rq2-conductor-cues/detector/uncle_annotations.json",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=REPOSITORY_ROOT / "rq2-conductor-cues/detector/transition_cue_pairing.v1.json",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_ROOT / "output/visual_all_cue_resets_v1/held_out_detector.json",
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
        default=PROJECT_ROOT / "output/visual_all_cue_resets_v1",
    )
    return parser.parse_args()


def condition_for_width(width_s: float) -> str:
    return f"automatic_all_cue_reset_width_{str(width_s).replace('.', '_')}s"


def build_landmarks(
    cues: list[dict],
    dense_live_t: np.ndarray,
    dense_ref_t: np.ndarray,
    *,
    reference_span_start_s: float,
    reference_frame_count: int,
) -> list[dict]:
    landmarks = []
    previous_reference_s = -np.inf
    for cue_index, cue in enumerate(cues):
        cue_time_s = float(cue["time"])
        reference_s = float(np.interp(cue_time_s, dense_live_t, dense_ref_t))
        if reference_s <= previous_reference_s:
            raise ValueError("cue landmarks are not strictly increasing in reference time")
        reference_frame = reference_frame_at_or_before(
            reference_s,
            span_start_s=reference_span_start_s,
            frame_rate=FRAME_RATE,
            frame_count=reference_frame_count,
        )
        landmarks.append(
            {
                "cue_index": cue_index,
                "cue_id": f"cue_{cue_index + 1:02d}",
                "cue_time_s": cue_time_s,
                "cue_label": cue["label"],
                "reference_s": reference_s,
                "reference_frame": reference_frame,
                "quantized_reference_s": (
                    reference_span_start_s + reference_frame / FRAME_RATE
                ),
            }
        )
        previous_reference_s = reference_s
    return landmarks


def select_eligible_landmark(
    current_reference_s: float,
    landmarks: list[dict],
    *,
    last_consumed_index: int,
    width_s: float,
) -> tuple[int | None, list[int]]:
    eligible = [
        index
        for index in range(last_consumed_index + 1, len(landmarks))
        if abs(float(landmarks[index]["reference_s"]) - current_reference_s)
        <= width_s + 1e-12
    ]
    if not eligible:
        return None, []
    selected = min(
        eligible,
        key=lambda index: (
            abs(float(landmarks[index]["reference_s"]) - current_reference_s),
            index,
        ),
    )
    return selected, eligible


def clear_follower(follower: object) -> None:
    if hasattr(follower, "input_features"):
        follower.input_features.clear()
    if hasattr(follower, "_alignment_path"):
        follower._alignment_path.clear()


def make_follower(reference_features: np.ndarray, offset: int):
    follower, _, _ = make_matchmaker_follower(
        reference_features[offset:],
        method=METHOD,
        frame_rate=FRAME_RATE,
    )
    return follower


def decode_with_oracle_landmarks(
    reference_features: np.ndarray,
    live_features: np.ndarray,
    availability_live_t: np.ndarray,
    landmarks: list[dict],
    *,
    reference_span_start_s: float,
) -> tuple[np.ndarray, list[dict]]:
    """Reset at every true cue occurrence and its dense-alignment landmark."""

    positions = np.empty(live_features.shape[0], dtype=np.float64)
    follower = make_follower(reference_features, 0)
    current_offset = 0
    last_relative_position = 0.0
    next_landmark = 0
    records: list[dict] = []

    for frame_index, available_s in enumerate(availability_live_t):
        while (
            next_landmark < len(landmarks)
            and float(landmarks[next_landmark]["cue_time_s"]) <= available_s + 1e-12
        ):
            landmark = landmarks[next_landmark]
            before_s = reference_span_start_s + last_relative_position
            clear_follower(follower)
            current_offset = int(landmark["reference_frame"])
            follower = make_follower(reference_features, current_offset)
            last_relative_position = current_offset / FRAME_RATE
            records.append(
                {
                    "cue_index": next_landmark,
                    "cue_id": landmark["cue_id"],
                    "cue_time_s": landmark["cue_time_s"],
                    "reset_available_s": float(available_s),
                    "reset_delay_s": float(available_s - landmark["cue_time_s"]),
                    "tracker_reference_before_s": before_s,
                    "reset_reference_s": landmark["quantized_reference_s"],
                }
            )
            next_landmark += 1

        if follower.is_still_following():
            relative = float(follower(live_features[frame_index], frame_index / FRAME_RATE))
            last_relative_position = current_offset / FRAME_RATE + relative
        positions[frame_index] = reference_span_start_s + last_relative_position

    clear_follower(follower)
    return positions, records


def decode_with_automatic_landmarks(
    reference_features: np.ndarray,
    live_features: np.ndarray,
    availability_live_t: np.ndarray,
    landmarks: list[dict],
    emissions: list[dict],
    *,
    width_s: float,
    reference_span_start_s: float,
) -> tuple[np.ndarray, dict]:
    """Apply detector emissions without using offline emission-to-cue matches."""

    positions = np.empty(live_features.shape[0], dtype=np.float64)
    follower = make_follower(reference_features, 0)
    current_offset = 0
    last_relative_position = 0.0
    last_consumed = -1
    next_emission = 0
    events: list[dict] = []
    selected_landmarks: list[int] = []
    skipped_landmarks: list[int] = []

    for frame_index, available_s in enumerate(availability_live_t):
        while (
            next_emission < len(emissions)
            and float(emissions[next_emission]["time_s"]) <= available_s + 1e-12
        ):
            emission = emissions[next_emission]
            current_reference_s = reference_span_start_s + last_relative_position
            selected, eligible = select_eligible_landmark(
                current_reference_s,
                landmarks,
                last_consumed_index=last_consumed,
                width_s=width_s,
            )
            record = {
                "emission_index": int(emission["emission_index"]),
                "emission_time_s": float(emission["time_s"]),
                "applied_at_audio_time_s": float(available_s),
                "audio_application_delay_s": float(available_s - emission["time_s"]),
                "tracker_reference_before_s": current_reference_s,
                "eligible_cue_indices": eligible,
                "selected_cue_index": selected,
                "selected_cue_id": landmarks[selected]["cue_id"] if selected is not None else None,
            }
            if selected is not None:
                newly_skipped = list(range(last_consumed + 1, selected))
                skipped_landmarks.extend(newly_skipped)
                landmark = landmarks[selected]
                clear_follower(follower)
                current_offset = int(landmark["reference_frame"])
                follower = make_follower(reference_features, current_offset)
                last_relative_position = current_offset / FRAME_RATE
                record["skipped_cue_indices"] = newly_skipped
                record["reset_reference_s"] = landmark["quantized_reference_s"]
                record["reference_jump_s"] = (
                    float(landmark["quantized_reference_s"]) - current_reference_s
                )
                selected_landmarks.append(selected)
                last_consumed = selected
            else:
                record["skipped_cue_indices"] = []
                record["reset_reference_s"] = None
                record["reference_jump_s"] = None
            events.append(record)
            next_emission += 1

        if follower.is_still_following():
            relative = float(follower(live_features[frame_index], frame_index / FRAME_RATE))
            last_relative_position = current_offset / FRAME_RATE + relative
        positions[frame_index] = reference_span_start_s + last_relative_position

    clear_follower(follower)
    remaining = list(range(last_consumed + 1, len(landmarks)))
    return positions, {
        "eligibility_width_s": width_s,
        "n_detector_emissions": len(emissions),
        "n_resets": len(selected_landmarks),
        "n_ignored_emissions": len(emissions) - len(selected_landmarks),
        "selected_cue_indices": selected_landmarks,
        "skipped_before_selected_cue_indices": skipped_landmarks,
        "remaining_unused_cue_indices": remaining,
        "events": events,
    }


def first_stable_reach_time(
    live_t: np.ndarray,
    reference_position_t: np.ndarray,
    target_reference_s: float,
    hold_s: float,
) -> float | None:
    """Return the first output confirming the full hold, without backdating."""
    reached = reference_position_t >= target_reference_s
    if hold_s <= 0:
        indices = np.flatnonzero(reached)
        return float(live_t[indices[0]]) if indices.size else None
    run_start_s = None
    for index, is_reached in enumerate(reached):
        if not is_reached:
            run_start_s = None
            continue
        current_s = float(live_t[index])
        if run_start_s is None:
            run_start_s = current_s
        # Allow floating-point rounding at an exact duration boundary.
        if current_s - run_start_s >= hold_s - 1e-10:
            return current_s
    return None


def transition_rows(
    transitions: list[Transition],
    live_t: np.ndarray,
    paths: dict[str, np.ndarray],
    *,
    paired_segments: set[str],
    hold_s: float,
) -> list[dict]:
    rows = []
    for transition in transitions:
        conditions = {}
        errors = {}
        for condition, path in paths.items():
            fire_s = first_stable_reach_time(
                live_t,
                path,
                transition.reference_target_s,
                hold_s,
            )
            error_s = fire_s - transition.live_target_s if fire_s is not None else None
            conditions[condition] = describe_transition(fire_s, error_s)
            errors[condition] = error_s
        row = {
            "segment": transition.segment,
            "included_primary": transition.segment not in PRIMARY_EXCLUSIONS,
            "has_approved_pair": transition.segment in paired_segments,
            "marked_live_target_s": transition.live_target_s,
            "reference_target_s": transition.reference_target_s,
            "conditions": conditions,
            "effects_relative_to_audio": {
                condition: classify_effect(errors["audio_only"], error_s)
                for condition, error_s in errors.items()
                if condition != "audio_only"
            },
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict], condition_names: list[str]) -> dict:
    def summarize(selected: list[dict]) -> dict:
        summary: dict = {"n_targets": len(selected), "conditions": {}}
        for condition in condition_names:
            values = [row["conditions"][condition]["signed_error_s"] for row in selected]
            usable = np.asarray([value for value in values if value is not None])
            summary["conditions"][condition] = {
                "n_reached": int(usable.size),
                "median_absolute_error_s": float(np.median(np.abs(usable))) if usable.size else None,
                "mean_absolute_error_s": float(np.mean(np.abs(usable))) if usable.size else None,
                "acceptable_count": int(
                    sum(row["conditions"][condition]["acceptable"] for row in selected)
                ),
                "within_limit_count": {
                    str(limit): int(
                        sum(
                            row["conditions"][condition]["within_limits"][str(limit)]
                            for row in selected
                        )
                    )
                    for limit in LIMITS_S
                },
            }
        summary["effects_relative_to_audio"] = {
            condition: {
                effect: int(
                    sum(
                        row["effects_relative_to_audio"].get(condition) == effect
                        for row in selected
                    )
                )
                for effect in sorted(
                    {
                        row["effects_relative_to_audio"].get(condition)
                        for row in selected
                        if row["effects_relative_to_audio"].get(condition) is not None
                    }
                )
            }
            for condition in condition_names
            if condition != "audio_only"
        }
        return summary

    return {
        "primary_14": summarize([row for row in rows if row["included_primary"]]),
        "sensitivity_all_16": summarize(rows),
        "approved_pairs_only_4": summarize(
            [row for row in rows if row["has_approved_pair"]]
        ),
    }


def assignment_audit(
    automatic_runs: dict[str, dict],
    detector: dict,
) -> dict:
    output = {}
    for condition, run in automatic_runs.items():
        by_tolerance = {}
        for tolerance, metrics in detector["event_metrics"].items():
            matched_by_fire = {
                int(item["fire_index"]): int(item["cue_index"])
                for item in metrics["matches"]
            }
            correct = wrong = matched_ignored = 0
            for event in run["events"]:
                emission = int(event["emission_index"])
                expected = matched_by_fire.get(emission)
                selected = event["selected_cue_index"]
                if expected is None:
                    continue
                if selected is None:
                    matched_ignored += 1
                elif selected == expected:
                    correct += 1
                else:
                    wrong += 1
            by_tolerance[tolerance] = {
                "offline_matched_emissions": len(matched_by_fire),
                "correct_landmark_resets": correct,
                "wrong_landmark_resets": wrong,
                "matched_emissions_ignored_by_gate": matched_ignored,
            }
        output[condition] = by_tolerance
    return output


def main() -> None:
    args = parse_args()
    required = (
        args.live,
        args.ref,
        args.reference,
        args.annotations,
        args.pairs,
        args.detector,
        args.timeline,
        args.approved_baseline_trace,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    trace_path = args.output_dir / "causal_paths.npz"
    landmark_path = args.output_dir / "cue_landmarks.json"
    if any(path.exists() for path in (result_path, trace_path, landmark_path)):
        raise SystemExit(f"refusing to overwrite all-cue output in {args.output_dir}")

    reference, dense_path, dense_live_t, dense_ref_t = load_dense_reference(args.reference)
    span = reference["shared_span"]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    cues = annotations["cues"]
    detector = json.loads(args.detector.read_text(encoding="utf-8"))
    emissions = detector["emissions"]
    pairs_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    if pairs_payload.get("status") != "approved_primary_pair_whitelist":
        raise RuntimeError("pairing file is not the approved descriptive whitelist")
    pairs = pairs_payload["primary_pair_whitelist"]
    transitions = parse_timeline(args.timeline, dense_live_t, dense_ref_t)

    from matchmaker.features.audio import LogSpectralEnergyProcessor

    started = time.perf_counter()
    processor = LogSpectralEnergyProcessor(sample_rate=SAMPLE_RATE, hop_length=HOP_SAMPLES)
    reference_audio = load_span_direct(
        args.ref,
        start=float(span["ref_start"]),
        end=float(span["ref_end"]),
        sr=SAMPLE_RATE,
    )
    reference_features = compute_matchmaker_features_chunked(
        reference_audio,
        processor=processor,
        hop=HOP_SAMPLES,
    )
    del reference_audio
    live_audio = load_span_direct(
        args.live,
        start=float(span["live_start"]),
        end=float(span["live_end"]),
        sr=SAMPLE_RATE,
    )
    live_features = compute_matchmaker_features_chunked(
        live_audio,
        processor=processor,
        hop=HOP_SAMPLES,
    )
    del live_audio
    feature_runtime_s = time.perf_counter() - started

    availability_live_t = (
        np.arange(live_features.shape[0], dtype=np.float64) / FRAME_RATE
        + HOP_SAMPLES / SAMPLE_RATE
        + float(span["live_start"])
    )
    landmarks = build_landmarks(
        cues,
        dense_live_t,
        dense_ref_t,
        reference_span_start_s=float(span["ref_start"]),
        reference_frame_count=reference_features.shape[0],
    )

    decode_started = time.perf_counter()
    baseline_relative, _, _, _ = decode_matchmaker(
        reference_features,
        live_features,
        method=METHOD,
        frame_rate=FRAME_RATE,
        progress=False,
    )
    paths_full: dict[str, np.ndarray] = {
        "audio_only": baseline_relative + float(span["ref_start"])
    }
    oracle_path, oracle_records = decode_with_oracle_landmarks(
        reference_features,
        live_features,
        availability_live_t,
        landmarks,
        reference_span_start_s=float(span["ref_start"]),
    )
    paths_full["perfect_all_cue_reset"] = oracle_path
    automatic_runs: dict[str, dict] = {}
    for width_s in ELIGIBILITY_WIDTHS_S:
        condition = condition_for_width(width_s)
        path, run = decode_with_automatic_landmarks(
            reference_features,
            live_features,
            availability_live_t,
            landmarks,
            emissions,
            width_s=width_s,
            reference_span_start_s=float(span["ref_start"]),
        )
        paths_full[condition] = path
        automatic_runs[condition] = run
    decode_runtime_s = time.perf_counter() - decode_started

    inside = (
        (availability_live_t >= dense_live_t[0])
        & (availability_live_t <= dense_live_t[-1])
    )
    live_t = availability_live_t[inside]
    paths = {name: path[inside] for name, path in paths_full.items()}
    ground_truth_ref_t = np.interp(live_t, dense_live_t, dense_ref_t)
    errors = {name: path - ground_truth_ref_t for name, path in paths.items()}

    agreement = baseline_agreement(
        live_t,
        paths["audio_only"],
        args.approved_baseline_trace,
    )
    if not agreement["same_time_grid"] or agreement[
        "max_absolute_reference_position_difference_s"
    ] > 1e-5:
        raise RuntimeError(f"audio baseline does not reproduce approved trace: {agreement}")

    paired_segments = {str(pair["segment"]) for pair in pairs}
    condition_names = list(paths)
    transition_sensitivity = {}
    for hold_s in STABILITY_HOLDS_S:
        rows = transition_rows(
            transitions,
            live_t,
            paths,
            paired_segments=paired_segments,
            hold_s=hold_s,
        )
        transition_sensitivity[str(hold_s)] = {
            "hold_s": hold_s,
            "rows": rows,
            "summary": summarize_rows(rows, condition_names),
        }

    primary_rows = transition_sensitivity["0.0"]["rows"]
    row_by_segment = {row["segment"]: row for row in primary_rows}
    pair_descriptions = []
    for pair in pairs:
        cue_index = int(str(pair["cue_id"]).split("_")[-1]) - 1
        pair_descriptions.append(
            {
                **pair,
                "cue_index": cue_index,
                "cue_minus_transition_target_s": (
                    float(pair["cue_time_s"]) - float(pair["target_time_s"])
                ),
                "transition_result": row_by_segment[str(pair["segment"])],
            }
        )

    landmark_payload = {
        "schema_version": 1,
        "status": "visual_contribution_all_cue_landmarks_frozen",
        "claim_limit": (
            "Reference positions come from the approved dense offline alignment. "
            "A deployable production would need equivalent score or rehearsal metadata."
        ),
        "landmarks": landmarks,
        "provenance": {
            "annotations": {
                "path": repository_path(args.annotations),
                "sha256": sha256(args.annotations),
            },
            "reference": {
                "path": repository_path(args.reference),
                "sha256": sha256(args.reference),
            },
            "dense_reference": {
                "path": repository_path(dense_path),
                "sha256": sha256(dense_path),
            },
        },
    }
    landmark_path.write_text(
        json.dumps(landmark_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = {
        "schema_version": 1,
        "status": "visual_contribution_all_cue_oracle_and_automatic_complete",
        "scope": "Firebird production-specific case study; all 22 cue landmarks",
        "claim_limit": (
            "The oracle supplies true cue occurrences. The automatic condition uses "
            "within-recording blocked cross-fit detector emissions. Neither result "
            "estimates performance on another conductor or production."
        ),
        "method": {
            "audio_tracker": METHOD,
            "implementation": "pymatchmaker",
            "version": MATCHMAKER_VERSION,
            "release_commit": MATCHMAKER_COMMIT,
            "frame_rate_hz": FRAME_RATE,
            "sample_rate_hz": SAMPLE_RATE,
            "hop_samples": HOP_SAMPLES,
            "eligibility_widths_reference_s": list(ELIGIBILITY_WIDTHS_S),
            "transition_stability_holds_s": list(STABILITY_HOLDS_S),
            "transition_fire_rule": (
                "first causal tracker output at or beyond each reference target; "
                "hold checks report the first output after the full elapsed hold, "
                "restarting the wait after any output below the target"
            ),
            "automatic_reset_rule": (
                "At each held-out detector emission, reset to the closest unused "
                "ordered cue landmark inside the fixed reference-position gate. "
                "Film-transition proximity and offline detector matching are not used."
            ),
            "acceptable_band_s": [ACCEPTABLE_EARLY_S, ACCEPTABLE_LATE_S],
            "absolute_error_limits_s": list(LIMITS_S),
            "primary_excluded_segments": sorted(PRIMARY_EXCLUSIONS),
        },
        "provenance": {
            "repository_revision_before_output": repository_revision(),
            "script": repository_path(Path(__file__)),
            "feature_runtime_s": feature_runtime_s,
            "decode_runtime_s": decode_runtime_s,
            "inputs": {
                "live_audio": external_media_record(args.live),
                "reference_audio": external_media_record(args.ref),
                "reference": {
                    "path": repository_path(args.reference),
                    "sha256": sha256(args.reference),
                },
                "dense_reference": {
                    "path": repository_path(dense_path),
                    "sha256": sha256(dense_path),
                },
                "annotations": {
                    "path": repository_path(args.annotations),
                    "sha256": sha256(args.annotations),
                },
                "descriptive_pairs": {
                    "path": repository_path(args.pairs),
                    "sha256": sha256(args.pairs),
                },
                "detector": {
                    "path": repository_path(args.detector),
                    "sha256": sha256(args.detector),
                },
                "timeline": {
                    "path": repository_path(args.timeline),
                    "sha256": sha256(args.timeline),
                },
                "approved_baseline_trace": {
                    "path": repository_path(args.approved_baseline_trace),
                    "sha256": sha256(args.approved_baseline_trace),
                },
            },
        },
        "baseline_reproduction_check": agreement,
        "oracle_resets": oracle_records,
        "automatic_runs": automatic_runs,
        "automatic_assignment_audit": assignment_audit(automatic_runs, detector),
        "approved_pair_descriptions": pair_descriptions,
        "dense_path_metrics": {
            name: dense_metrics(error) for name, error in errors.items()
        },
        "path_diagnostics": {
            name: path_diagnostics(path) for name, path in paths.items()
        },
        "transition_sensitivity": transition_sensitivity,
        "transition_summary": transition_sensitivity["0.0"]["summary"],
    }

    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_result.replace(result_path)
    temporary_trace = trace_path.with_suffix(".npz.tmp")
    with temporary_trace.open("wb") as handle:
        arrays = {
            "online_live_t": live_t,
            "ground_truth_ref_t": ground_truth_ref_t,
        }
        for name, path in paths.items():
            arrays[f"{name}_ref_t"] = path
            arrays[f"{name}_error_s"] = errors[name]
        np.savez_compressed(handle, **arrays)
    temporary_trace.replace(trace_path)
    print(json.dumps(payload["transition_summary"]["primary_14"], indent=2))
    print(f"wrote {landmark_path}")
    print(f"wrote {result_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
