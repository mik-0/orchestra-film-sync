#!/usr/bin/env python3
"""Reproduce the thesis-facing cue-characterisation feature and cue-cluster audit."""

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


from collections.abc import Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import combined_features as cluster_features


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_FEATURE_INPUT = (
    REPOSITORY_ROOT
    / "rq2-conductor-cues/features/analysis_results/analysis_v2.json"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "analysis_results/cue_characterisation_analysis.json"

FEATURE_FAMILIES = {
    "Combined hand speed": "combined_speed_smooth",
    "Upward velocity": "upward_vel_smooth",
    "Stillness-burst": "stillness_burst_smooth",
    "Positive acceleration": "positive_accel_smooth",
    "Jerk (speed change rate)": "jerk_smooth",
    "Speed surge": "surge_ratio_smooth",
    "Spread opening": "spread_opening_smooth",
    "Speed asymmetry": "speed_asymmetry",
}
SUMMARY_STATISTICS = ("max", "imm", "pre", "post", "mean")
RANDOM_SEED = 20260828


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_FEATURE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def cohens_d(values: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    cue = values[mask & (labels == 1)]
    noncue = values[mask & (labels == 0)]
    pooled = math.sqrt(
        (
            (len(cue) - 1) * cue.var(ddof=1)
            + (len(noncue) - 1) * noncue.var(ddof=1)
        )
        / (len(cue) + len(noncue) - 2)
    )
    return float((cue.mean() - noncue.mean()) / (pooled + 1e-12))


def pairwise_auc(values: np.ndarray, labels: np.ndarray, direction: int = 1) -> float:
    cue = values[labels == 1]
    noncue = values[labels == 0]
    score = 0.0
    for cue_value in cue:
        for noncue_value in noncue:
            difference = direction * (cue_value - noncue_value)
            score += 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
    return float(score / (len(cue) * len(noncue)))


def feature_audit(feature_input: Path) -> dict:
    source = json.loads(feature_input.read_text(encoding="utf-8"))
    labels = np.asarray(source["labels"], dtype=int)
    rows = source["transitions"]
    all_rows = np.ones(len(labels), dtype=bool)
    cue_indices = np.flatnonzero(labels == 1)
    noncue_indices = np.flatnonzero(labels == 0)

    results = []
    for family, base_key in FEATURE_FAMILIES.items():
        candidates = []
        for statistic in SUMMARY_STATISTICS:
            key = f"{base_key}_{statistic}"
            values = np.asarray([float(row[key]) for row in rows])
            effect = cohens_d(values, labels, all_rows)
            candidates.append((statistic, values, effect))

        statistic, values, effect = max(candidates, key=lambda item: abs(item[2]))
        direction = 1 if effect >= 0 else -1
        raw_auc = pairwise_auc(values, labels)
        oriented_auc = pairwise_auc(values, labels, direction)

        held_pair_score = 0.0
        selected_statistics: dict[str, int] = {
            name: 0 for name in SUMMARY_STATISTICS
        }
        for cue_index in cue_indices:
            for noncue_index in noncue_indices:
                train = np.ones(len(labels), dtype=bool)
                train[[cue_index, noncue_index]] = False
                training_candidates = []
                for candidate_statistic, candidate_values, _ in candidates:
                    training_effect = cohens_d(candidate_values, labels, train)
                    training_candidates.append(
                        (
                            abs(training_effect),
                            candidate_statistic,
                            1 if training_effect >= 0 else -1,
                            candidate_values,
                        )
                    )
                _, selected, selected_direction, selected_values = max(
                    training_candidates, key=lambda item: item[0]
                )
                selected_statistics[selected] += 1
                difference = selected_direction * (
                    selected_values[cue_index] - selected_values[noncue_index]
                )
                held_pair_score += (
                    1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
                )

        results.append(
            {
                "feature": family,
                "selected_statistic_full_data": statistic,
                "cohens_d_full_data": effect,
                "auc_raw_direction_full_data": raw_auc,
                "auc_oriented_full_data": oriented_auc,
                "selection_aware_held_pair_auc": float(
                    held_pair_score / (len(cue_indices) * len(noncue_indices))
                ),
                "held_pair_training_selection_counts": selected_statistics,
            }
        )

    return {
        "scope": (
            "Eight author-labelled clear-cue transition targets versus eight "
            "author-labelled non-cue transition targets in Firebird."
        ),
        "n_cue_targets": int((labels == 1).sum()),
        "n_noncue_targets": int((labels == 0).sum()),
        "candidate_statistics_per_feature": list(SUMMARY_STATISTICS),
        "selection_aware_check": (
            "Leave one cue target and one non-cue target out, select the summary "
            "statistic and direction on the remaining 14 targets, and score the "
            "held-out pair. Average over all 64 cue/non-cue pairs."
        ),
        "features": results,
    }


def cue_feature_matrix() -> tuple[np.ndarray, list[str], list[dict]]:
    cache = np.load(cluster_features.CACHE_FILE)
    keypoints = cache["keypoints"]
    fps = float(cache["fps"])
    total_frames = int(cache["total_frames"])
    cues = json.loads(
        cluster_features.ANNOT_FILE.read_text(encoding="utf-8")
    )["cues"]

    feature_names: list[str] = []
    rows = []
    for cue_index, cue in enumerate(cues):
        row = {}
        for window_name, start_s, end_s in cluster_features.WINDOWS:
            values = cluster_features.window_features(
                keypoints, cue["time"], fps, total_frames, start_s, end_s
            )
            if values is None:
                raise RuntimeError(
                    f"cue {cue_index} has no data for window {window_name}"
                )
            for key, value in values.items():
                row[f"{window_name}__{key}"] = value
        if cue_index == 0:
            feature_names = list(row)
        if set(row) != set(feature_names):
            raise RuntimeError("inconsistent cue-cluster feature columns")
        rows.append([row[name] for name in feature_names])
    return np.asarray(rows, dtype=float), feature_names, cues


def ward_partition(matrix: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    standard_deviation = matrix.std(axis=0)
    keep = standard_deviation >= 1e-8
    standardised = (
        matrix[:, keep] - matrix[:, keep].mean(axis=0)
    ) / standard_deviation[keep]
    pca = PCA(svd_solver="full")
    scores = pca.fit_transform(standardised)
    n_components = int(
        np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1
    )
    reduced = scores[:, :n_components]
    partition = fcluster(
        linkage(pdist(reduced), method="ward"),
        t=n_clusters,
        criterion="maxclust",
    )
    return (
        partition,
        reduced,
        pca.explained_variance_ratio_,
        n_components,
    )


def cluster_sizes(labels: Sequence[int]) -> list[int]:
    return sorted(
        (int(count) for count in np.bincount(np.asarray(labels))[1:] if count),
        reverse=True,
    )


def cluster_audit() -> dict:
    matrix, feature_names, cues = cue_feature_matrix()
    musical_labels = [cue["label"] for cue in cues]
    full_partition, reduced, explained, n_components = ward_partition(matrix, 5)

    solutions = []
    for n_clusters in range(2, 8):
        partition, cluster_space, _, _ = ward_partition(matrix, n_clusters)
        solutions.append(
            {
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes(partition),
                "silhouette": float(silhouette_score(cluster_space, partition)),
                "adjusted_rand_index_vs_musical_labels": float(
                    adjusted_rand_score(musical_labels, partition)
                ),
            }
        )

    leave_one_cue_out = []
    for cue_index in range(len(cues)):
        keep = np.arange(len(cues)) != cue_index
        partition, _, _, _ = ward_partition(matrix[keep], 5)
        leave_one_cue_out.append(
            float(adjusted_rand_score(full_partition[keep], partition))
        )

    leave_one_window_out = {}
    for window_name, _, _ in cluster_features.WINDOWS:
        keep = np.asarray(
            [not name.startswith(f"{window_name}__") for name in feature_names]
        )
        partition, _, _, _ = ward_partition(matrix[:, keep], 5)
        leave_one_window_out[window_name] = {
            "adjusted_rand_index_vs_full_five_group_cut": float(
                adjusted_rand_score(full_partition, partition)
            ),
            "cluster_sizes": cluster_sizes(partition),
        }

    random = np.random.default_rng(RANDOM_SEED)
    feature_subsample_ari = []
    n_selected_features = round(0.80 * matrix.shape[1])
    for _ in range(500):
        selected = np.sort(
            random.choice(
                matrix.shape[1], size=n_selected_features, replace=False
            )
        )
        partition, _, _, _ = ward_partition(matrix[:, selected], 5)
        feature_subsample_ari.append(
            float(adjusted_rand_score(full_partition, partition))
        )

    return {
        "scope": "Twenty-two author-annotated Firebird cues.",
        "n_cues": len(cues),
        "n_features": int(matrix.shape[1]),
        "windows": [
            {"name": name, "start_s": start_s, "end_s": end_s}
            for name, start_s, end_s in cluster_features.WINDOWS
        ],
        "pca": {
            "components_for_at_least_90_percent_variance": n_components,
            "first_four_explained_variance_ratios": [
                float(value) for value in explained[:4]
            ],
        },
        "solutions": solutions,
        "displayed_five_group_cut": {
            "cluster_sizes": cluster_sizes(full_partition),
            "interpretation": (
                "An illustrative cut showing three multi-cue groups and two "
                "single-cue outliers, not an optimised or validated taxonomy."
            ),
        },
        "stability": {
            "leave_one_cue_out_ari": {
                "mean": float(np.mean(leave_one_cue_out)),
                "median": float(np.median(leave_one_cue_out)),
                "minimum": float(np.min(leave_one_cue_out)),
                "maximum": float(np.max(leave_one_cue_out)),
            },
            "leave_one_window_out": leave_one_window_out,
            "random_80_percent_feature_subsamples": {
                "seed": RANDOM_SEED,
                "iterations": 500,
                "mean_ari": float(np.mean(feature_subsample_ari)),
                "quantiles": {
                    str(quantile): float(np.quantile(feature_subsample_ari, quantile))
                    for quantile in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
                },
            },
        },
    }


def main() -> None:
    args = parse_args()
    for required in (
        args.feature_input,
        cluster_features.CACHE_FILE,
        cluster_features.ANNOT_FILE,
    ):
        if not required.is_file():
            raise SystemExit(f"missing required cue-characterisation input: {required}")

    payload = {
        "schema_version": 1,
        "status": "cue_characterisation_feature_and_cluster_audit_complete",
        "scope": (
            "Firebird cue-characterisation only. This audit does not claim another conductor or "
            "performance."
        ),
        "feature_separation": feature_audit(args.feature_input),
        "cue_clustering": cluster_audit(),
        "provenance": {
            "script": repository_path(Path(__file__)),
            "inputs": {
                repository_path(args.feature_input): sha256(args.feature_input),
                repository_path(cluster_features.CACHE_FILE): sha256(
                    cluster_features.CACHE_FILE
                ),
                repository_path(cluster_features.ANNOT_FILE): sha256(
                    cluster_features.ANNOT_FILE
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
