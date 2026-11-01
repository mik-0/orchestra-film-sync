#!/usr/bin/env python3
"""Export the thesis-facing visual and transition-score summary."""

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
EARLY_LIMIT_S = 0.60
LATE_LIMIT_S = 0.33
CATASTROPHE_S = 1.0
CATASTROPHE_SURCHARGE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_INPUT.parent / "thesis_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_INPUT.parent / "thesis_summary.csv",
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


def transition_cost(error_s: float) -> float:
    """Return the asymmetric, catastrophe-aware cost defined in Chapter 5."""

    if -EARLY_LIMIT_S <= error_s <= LATE_LIMIT_S:
        return 0.0
    if abs(error_s) >= CATASTROPHE_S:
        return 1.0 + CATASTROPHE_SURCHARGE
    side_limit = EARLY_LIMIT_S if error_s < 0 else LATE_LIMIT_S
    return (abs(error_s) - side_limit) / (CATASTROPHE_S - side_limit)


def summarize_rows(rows: list[dict], condition_names: list[str]) -> dict:
    summary = {"n_targets": len(rows), "conditions": {}}
    for condition in condition_names:
        errors = [float(row["conditions"][condition]["signed_error_s"]) for row in rows]
        summary["conditions"][condition] = {
            "acceptable_count": sum(-EARLY_LIMIT_S <= error <= LATE_LIMIT_S for error in errors),
            "median_absolute_error_s": statistics.median(abs(error) for error in errors),
            "mean_transition_cost": statistics.mean(transition_cost(error) for error in errors),
            "catastrophe_count": sum(abs(error) >= CATASTROPHE_S for error in errors),
            "within_absolute_error_count": {
                "0.5": sum(abs(error) <= 0.5 for error in errors),
                "1.0": sum(abs(error) <= 1.0 for error in errors),
                "2.0": sum(abs(error) <= 2.0 for error in errors),
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing visual-contribution result: {args.input}")
    for output in (args.output_json, args.output_csv):
        if output.exists():
            raise SystemExit(f"refusing to overwrite summary output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("status") != "visual_contribution_all_cue_oracle_and_automatic_complete":
        raise RuntimeError("input is not a completed all-cue visual-contribution result")
    one_shot = source["transition_sensitivity"]["0.0"]
    rows = one_shot["rows"]
    condition_names = list(rows[0]["conditions"])
    primary_rows = [row for row in rows if row["included_primary"]]

    stability = {}
    for hold_key, view in source["transition_sensitivity"].items():
        stability[hold_key] = {
            condition: view["summary"]["primary_14"]["conditions"][condition][
                "acceptable_count"
            ]
            for condition in condition_names
        }

    payload = {
        "schema_version": 1,
        "status": "visual_contribution_thesis_summary_complete",
        "source": {
            "path": repository_path(args.input),
            "sha256": sha256(args.input),
            "source_status": source["status"],
        },
        "score": {
            "acceptable_band_s": [-EARLY_LIMIT_S, LATE_LIMIT_S],
            "catastrophe_threshold_s": CATASTROPHE_S,
            "catastrophe_surcharge": CATASTROPHE_SURCHARGE,
        },
        "one_shot": {
            "primary_14": summarize_rows(primary_rows, condition_names),
            "sensitivity_all_16": summarize_rows(rows, condition_names),
        },
        "primary_acceptable_count_by_stability_hold_s": stability,
    }

    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scope",
            "condition",
            "n_targets",
            "acceptable_count",
            "median_absolute_error_s",
            "mean_transition_cost",
            "catastrophe_count",
            "within_0_5s_count",
            "within_1_0s_count",
            "within_2_0s_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for scope, scope_summary in payload["one_shot"].items():
            for condition, values in scope_summary["conditions"].items():
                writer.writerow(
                    {
                        "scope": scope,
                        "condition": condition,
                        "n_targets": scope_summary["n_targets"],
                        "acceptable_count": values["acceptable_count"],
                        "median_absolute_error_s": values["median_absolute_error_s"],
                        "mean_transition_cost": values["mean_transition_cost"],
                        "catastrophe_count": values["catastrophe_count"],
                        "within_0_5s_count": values["within_absolute_error_count"]["0.5"],
                        "within_1_0s_count": values["within_absolute_error_count"]["1.0"],
                        "within_2_0s_count": values["within_absolute_error_count"]["2.0"],
                    }
                )

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
