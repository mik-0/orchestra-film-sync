#!/usr/bin/env python3
"""Test the transition-score conclusion across catastrophe parameters."""

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
import hashlib
import json
from pathlib import Path
import statistics


PROJECT_ROOT = _AUDIO_ROOT
REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_INPUT = PROJECT_ROOT / "output/visual_all_cue_resets_v1/results.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "output/transition_score_sensitivity.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "output/transition_score_sensitivity.csv"
EARLY_LIMIT_S = 0.60
LATE_LIMIT_S = 0.33
CATASTROPHE_THRESHOLDS_S = (0.75, 1.0, 1.5, 2.0)
CATASTROPHE_SURCHARGES = (0.0, 0.5, 1.0, 2.0)
BASELINE = "audio_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing output files",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def transition_cost(
    error_s: float,
    catastrophe_threshold_s: float,
    catastrophe_surcharge: float,
    early_limit_s: float = EARLY_LIMIT_S,
    late_limit_s: float = LATE_LIMIT_S,
) -> float:
    """Return the thesis transition cost for one signed firing error."""

    if catastrophe_threshold_s <= max(early_limit_s, late_limit_s):
        raise ValueError("catastrophe threshold must exceed both tolerance limits")
    if catastrophe_surcharge < 0:
        raise ValueError("catastrophe surcharge must be non-negative")
    if -early_limit_s <= error_s <= late_limit_s:
        return 0.0
    if abs(error_s) >= catastrophe_threshold_s:
        return 1.0 + catastrophe_surcharge
    side_limit = early_limit_s if error_s < 0 else late_limit_s
    return (abs(error_s) - side_limit) / (
        catastrophe_threshold_s - side_limit
    )


def summarize_condition(
    rows: list[dict],
    condition: str,
    catastrophe_threshold_s: float,
    catastrophe_surcharge: float,
) -> dict:
    errors = [float(row["conditions"][condition]["signed_error_s"]) for row in rows]
    costs = [
        transition_cost(error, catastrophe_threshold_s, catastrophe_surcharge)
        for error in errors
    ]
    return {
        "n_targets": len(errors),
        "mean_cost": statistics.mean(costs),
        "median_absolute_error_s": statistics.median(abs(error) for error in errors),
        "catastrophe_count": sum(
            abs(error) >= catastrophe_threshold_s for error in errors
        ),
    }


def analyze_scope(rows: list[dict], condition_names: list[str]) -> dict:
    grid = []
    deltas: dict[str, list[float]] = {
        condition: [] for condition in condition_names if condition != BASELINE
    }
    for threshold_s in CATASTROPHE_THRESHOLDS_S:
        for surcharge in CATASTROPHE_SURCHARGES:
            conditions = {
                condition: summarize_condition(
                    rows,
                    condition,
                    threshold_s,
                    surcharge,
                )
                for condition in condition_names
            }
            baseline_cost = conditions[BASELINE]["mean_cost"]
            for condition, values in conditions.items():
                values["mean_cost_difference_from_audio_only"] = (
                    values["mean_cost"] - baseline_cost
                )
                if condition != BASELINE:
                    deltas[condition].append(
                        values["mean_cost_difference_from_audio_only"]
                    )
            grid.append(
                {
                    "catastrophe_threshold_s": threshold_s,
                    "catastrophe_surcharge": surcharge,
                    "conditions": conditions,
                }
            )

    comparisons = {}
    for condition, values in deltas.items():
        comparisons[condition] = {
            "lower_cost_than_audio_only_cells": sum(value < 0 for value in values),
            "equal_cost_to_audio_only_cells": sum(value == 0 for value in values),
            "higher_cost_than_audio_only_cells": sum(value > 0 for value in values),
            "minimum_mean_cost_difference": min(values),
            "maximum_mean_cost_difference": max(values),
        }
    return {
        "n_targets": len(rows),
        "grid": grid,
        "comparisons_to_audio_only": comparisons,
    }


def write_csv(path: Path, scopes: dict) -> None:
    fieldnames = [
        "scope",
        "catastrophe_threshold_s",
        "catastrophe_surcharge",
        "condition",
        "n_targets",
        "mean_cost",
        "mean_cost_difference_from_audio_only",
        "median_absolute_error_s",
        "catastrophe_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for scope_name, scope in scopes.items():
            for cell in scope["grid"]:
                for condition, values in cell["conditions"].items():
                    writer.writerow(
                        {
                            "scope": scope_name,
                            "catastrophe_threshold_s": cell[
                                "catastrophe_threshold_s"
                            ],
                            "catastrophe_surcharge": cell[
                                "catastrophe_surcharge"
                            ],
                            "condition": condition,
                            **values,
                        }
                    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing visual-contribution result: {args.input}")
    for output in (args.output_json, args.output_csv):
        if output.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite output without --force: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("status") != "visual_contribution_all_cue_oracle_and_automatic_complete":
        raise RuntimeError("input is not a completed all-cue visual-contribution result")
    rows = source["transition_sensitivity"]["0.0"]["rows"]
    condition_names = list(rows[0]["conditions"])
    if BASELINE not in condition_names:
        raise RuntimeError("input has no audio-only baseline")
    primary_rows = [row for row in rows if row["included_primary"]]

    scopes = {
        "primary_valid_14": analyze_scope(primary_rows, condition_names),
        "sensitivity_all_16": analyze_scope(rows, condition_names),
    }
    payload = {
        "schema_version": 1,
        "status": "transition_score_parameter_sensitivity_complete",
        "purpose": (
            "Test whether condition rankings depend on the declared catastrophe "
            "threshold and surcharge. The director-derived tolerance band is fixed."
        ),
        "script": repository_path(Path(__file__)),
        "source": {
            "path": repository_path(args.input),
            "sha256": sha256(args.input),
            "source_status": source["status"],
            "transition_view": "one-shot target crossing (hold_s=0.0)",
        },
        "fixed_acceptable_band_s": [-EARLY_LIMIT_S, LATE_LIMIT_S],
        "catastrophe_thresholds_s": list(CATASTROPHE_THRESHOLDS_S),
        "catastrophe_surcharges": list(CATASTROPHE_SURCHARGES),
        "scopes": scopes,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, scopes)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(json.dumps(scopes["primary_valid_14"]["comparisons_to_audio_only"], indent=2))


if __name__ == "__main__":
    main()
