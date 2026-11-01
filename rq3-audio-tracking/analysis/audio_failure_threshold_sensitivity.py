"""Report audio-failure accuracy and sustained failures at several error limits."""

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
import sys

import numpy as np


HERE = _AUDIO_ROOT
REPOSITORY = _REPO_ROOT
sys.path.insert(0, str(HERE))

from online_dtw.alignment_reference import failure_episodes  # noqa: E402


RESULTS = (
    "published_firebird_matchmaker_dixon.json",
    "published_chopin_matchmaker_dixon.json",
    "published_yuja_fedorova_matchmaker_dixon.json",
    "published_daphnis_matchmaker_dixon.json",
)
ALIGNMENT_LIMITS_S = (0.05, 0.10, 0.50, 1.0, 2.0)
FAILURE_LIMITS_S = (0.50, 1.0, 2.0)
MINIMUM_FAILURE_DURATION_S = 1.0
LOW_ENERGY_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=HERE / "output" / "audio_failure_threshold_sensitivity.json",
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=HERE / "output" / "audio_failure_alignment_rates.csv",
    )
    parser.add_argument(
        "--failure-csv",
        type=Path,
        default=HERE / "output" / "audio_failure_threshold_sensitivity.csv",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY))


def alignment_rates(error_s: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error_s[np.isfinite(error_s)])
    return {
        str(limit): float(np.mean(absolute <= limit))
        for limit in ALIGNMENT_LIMITS_S
    }


def starts_after_low_energy(
    episode: dict,
    sample_t: np.ndarray,
    energy_ratio: np.ndarray,
) -> bool:
    before = (
        (sample_t >= float(episode["live_start"]) - 1.0)
        & (sample_t < float(episode["live_start"]))
    )
    return bool(
        np.any(before)
        and float(np.mean(energy_ratio[before])) <= LOW_ENERGY_RATIO
    )


def failure_summary(
    live_t: np.ndarray,
    error_s: np.ndarray,
    sample_t: np.ndarray,
    energy_ratio: np.ndarray,
    *,
    threshold_s: float,
    shared_duration_s: float,
) -> dict:
    episodes = failure_episodes(
        live_t,
        error_s,
        threshold_s=threshold_s,
        minimum_duration_s=MINIMUM_FAILURE_DURATION_S,
    )
    durations = np.asarray(
        [episode["duration_s"] for episode in episodes], dtype=float
    )
    behind = sum(episode["peak_error_s"] < 0 for episode in episodes)
    low_energy = sum(
        starts_after_low_energy(episode, sample_t, energy_ratio)
        for episode in episodes
    )
    failure_duration_s = float(np.sum(durations))
    return {
        "threshold_s": threshold_s,
        "minimum_duration_s": MINIMUM_FAILURE_DURATION_S,
        "episode_count": len(episodes),
        "failure_duration_s": failure_duration_s,
        "failure_time_fraction": failure_duration_s / shared_duration_s,
        "median_episode_duration_s": (
            float(np.median(durations)) if durations.size else None
        ),
        "peak_behind_count": behind,
        "peak_ahead_count": len(episodes) - behind,
        "peak_behind_fraction": behind / len(episodes) if episodes else None,
        "episodes_starting_after_low_energy": low_energy,
    }


def main() -> None:
    args = parse_args()
    recordings = []
    for filename in RESULTS:
        result_path = HERE / "output" / filename
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "final":
            raise ValueError(f"result is not final: {result_path}")
        trace_path = REPOSITORY / result["provenance"]["trace_path"]
        with np.load(trace_path) as trace:
            live_t = trace["online_live_t"].astype(float)
            error_s = trace["error_s"].astype(float)
            sample_t = trace["sample_t"].astype(float)
            energy_ratio = trace["sample_energy_ratio"].astype(float)
        span = result["reference"]["shared_span"]
        duration_s = float(span["live_end"] - span["live_start"])
        recordings.append({
            "pair": result["pair"],
            "duration_s": duration_s,
            "alignment_rates": alignment_rates(error_s),
            "sustained_failures": [
                failure_summary(
                    live_t,
                    error_s,
                    sample_t,
                    energy_ratio,
                    threshold_s=limit,
                    shared_duration_s=duration_s,
                )
                for limit in FAILURE_LIMITS_S
            ],
            "inputs": {
                "result": {
                    "path": repository_path(result_path),
                    "sha256": sha256(result_path),
                },
                "trace": {
                    "path": repository_path(trace_path),
                    "sha256": sha256(trace_path),
                },
            },
        })

    aggregate_failures = []
    for limit in FAILURE_LIMITS_S:
        rows = [
            next(
                item for item in recording["sustained_failures"]
                if item["threshold_s"] == limit
            )
            for recording in recordings
        ]
        episode_count = sum(row["episode_count"] for row in rows)
        behind_count = sum(row["peak_behind_count"] for row in rows)
        failure_duration_s = sum(row["failure_duration_s"] for row in rows)
        total_duration_s = sum(recording["duration_s"] for recording in recordings)
        aggregate_failures.append({
            "threshold_s": limit,
            "minimum_duration_s": MINIMUM_FAILURE_DURATION_S,
            "episode_count": episode_count,
            "failure_duration_s": failure_duration_s,
            "duration_weighted_failure_fraction": (
                failure_duration_s / total_duration_s
            ),
            "peak_behind_count": behind_count,
            "peak_ahead_count": episode_count - behind_count,
            "peak_behind_fraction": behind_count / episode_count,
            "episodes_starting_after_low_energy": sum(
                row["episodes_starting_after_low_energy"] for row in rows
            ),
        })

    payload = {
        "status": "final",
        "alignment_rate_limits_s": ALIGNMENT_LIMITS_S,
        "sustained_failure_limits_s": FAILURE_LIMITS_S,
        "minimum_failure_duration_s": MINIMUM_FAILURE_DURATION_S,
        "recordings_are_reporting_units": True,
        "recordings": recordings,
        "aggregate_sustained_failures": aggregate_failures,
        "interpretation": (
            "Thresholds change the amount labelled as failure, but the "
            "predominantly-behind direction and weak low-energy count persist."
        ),
        "provenance": {
            "script": repository_path(Path(__file__)),
            "script_sha256": sha256(Path(__file__)),
        },
    }

    for path in (args.output_json, args.alignment_csv, args.failure_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with args.alignment_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pair", "limit_s", "rate_within_limit"),
        )
        writer.writeheader()
        for recording in recordings:
            for limit, rate in recording["alignment_rates"].items():
                writer.writerow({
                    "pair": recording["pair"],
                    "limit_s": limit,
                    "rate_within_limit": rate,
                })

    with args.failure_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(aggregate_failures[0]),
        )
        writer.writeheader()
        writer.writerows(aggregate_failures)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.alignment_csv}")
    print(f"wrote {args.failure_csv}")


if __name__ == "__main__":
    main()
