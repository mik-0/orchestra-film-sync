#!/usr/bin/env python3
"""Export an out-of-sample cue-detector trace for the visual-contribution reset experiment.

Five contiguous, cue-balanced time blocks are scored by models that were fitted
without that block. Feature selection is repeated inside each training fold.
The exported detector emissions are later consumed causally by the audio reset
runner. Offline cue matching is included only for detector evaluation.
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
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from live_detector import (
    ANNOT,
    CACHE,
    NEG_MARGIN_SEC,
    NEG_PER_SEC,
    POS_TOL_SEC,
    WARMUP_SEC,
    _all_rich_names,
    causal_smooth,
    detect_peaks,
    extract_features_rich,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "shared/audio/output/visual_all_cue_resets_v1"
)
N_FOLDS = 5
N_SELECTED_FEATURES = 120
PURGE_SEC = 2.5
OPERATING_THRESHOLD = 0.60
EVENT_TOLERANCES_S = (0.12, 0.20, 0.48, 1.00)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--annotations", type=Path, default=ANNOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD)
    return parser.parse_args()


def cue_balanced_folds(
    cue_frames: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    n_folds: int,
) -> list[tuple[int, int]]:
    """Put similar cue counts in contiguous folds with boundaries between cues."""

    if n_folds < 2 or n_folds > len(cue_frames):
        raise ValueError("n_folds must be between 2 and the number of cues")
    groups = np.array_split(np.sort(cue_frames), n_folds)
    edges = [start_frame]
    for left, right in zip(groups[:-1], groups[1:]):
        edges.append(int((int(left[-1]) + int(right[0])) // 2))
    edges.append(end_frame)
    folds = list(zip(edges[:-1], edges[1:]))
    if any(stop <= start for start, stop in folds):
        raise ValueError("fold construction produced an empty block")
    return folds


def build_sample_index(
    cue_frames: np.ndarray,
    *,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic positive and negative frame labels."""

    positive_radius = int(round(POS_TOL_SEC * fps))
    frames: list[int] = []
    labels: list[int] = []
    cue_ids: list[int] = []
    for cue_id, cue_frame in enumerate(cue_frames):
        for offset in range(-positive_radius, positive_radius + 1):
            frame = int(cue_frame + offset)
            if start_frame <= frame < end_frame:
                frames.append(frame)
                labels.append(1)
                cue_ids.append(cue_id)

    skip = max(1, int(round(fps / NEG_PER_SEC)))
    negative_margin = int(round(NEG_MARGIN_SEC * fps))
    for frame in range(start_frame, end_frame, skip):
        if np.min(np.abs(cue_frames - frame)) > negative_margin:
            frames.append(frame)
            labels.append(0)
            cue_ids.append(-1)

    order = np.argsort(frames, kind="stable")
    return (
        np.asarray(frames, dtype=np.int64)[order],
        np.asarray(labels, dtype=np.int8)[order],
        np.asarray(cue_ids, dtype=np.int64)[order],
    )


def select_features(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_features: int,
) -> np.ndarray:
    """Fit the LDA ranking on training data only."""

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(scaled, labels)
    ranking = np.argsort(-np.abs(lda.coef_[0]))
    return ranking[: min(n_features, features.shape[1])].astype(np.int64)


def chronological_event_match(
    cue_times_s: np.ndarray,
    fire_times_s: np.ndarray,
    tolerance_s: float,
) -> list[tuple[int, int]]:
    """Maximise ordered matches, then minimise their total absolute offset."""

    cues = tuple(float(value) for value in cue_times_s)
    fires = tuple(float(value) for value in fire_times_s)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if i >= len(cues) or j >= len(fires):
            return 0, 0.0, ()
        candidates = [solve(i + 1, j), solve(i, j + 1)]
        offset = abs(cues[i] - fires[j])
        if offset <= tolerance_s + 1e-12:
            count, cost, pairs = solve(i + 1, j + 1)
            candidates.append((count + 1, cost + offset, ((i, j),) + pairs))
        return max(candidates, key=lambda item: (item[0], -item[1]))

    return list(solve(0, 0)[2])


def event_metrics(
    cue_times_s: np.ndarray,
    fire_times_s: np.ndarray,
    tolerance_s: float,
) -> dict:
    pairs = chronological_event_match(cue_times_s, fire_times_s, tolerance_s)
    matched_cues = {cue for cue, _ in pairs}
    matched_fires = {fire for _, fire in pairs}
    offsets = [float(fire_times_s[j] - cue_times_s[i]) for i, j in pairs]
    precision = len(pairs) / len(fire_times_s) if len(fire_times_s) else 0.0
    recall = len(pairs) / len(cue_times_s) if len(cue_times_s) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_s": tolerance_s,
        "matched_count": len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unmatched_cue_indices": sorted(set(range(len(cue_times_s))) - matched_cues),
        "unmatched_fire_indices": sorted(set(range(len(fire_times_s))) - matched_fires),
        "signed_emission_offsets_s": offsets,
        "mean_signed_emission_offset_s": float(np.mean(offsets)) if offsets else None,
        "mean_absolute_emission_offset_s": (
            float(np.mean(np.abs(offsets))) if offsets else None
        ),
        "matches": [
            {
                "cue_index": cue,
                "fire_index": fire,
                "signed_offset_s": float(fire_times_s[fire] - cue_times_s[cue]),
            }
            for cue, fire in pairs
        ],
    }


def main() -> None:
    args = parse_args()
    for path in (args.cache, args.annotations):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("threshold must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "held_out_detector.json"
    trace_path = args.output_dir / "held_out_detector_trace.npz"
    if result_path.exists() or trace_path.exists():
        raise SystemExit(f"refusing to overwrite detector output in {args.output_dir}")

    cache = np.load(args.cache)
    keypoints = cache["keypoints"]
    fps = float(cache["fps"])
    total_frames = int(cache["total_frames"])
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    cues = annotations["cues"]
    cue_times_s = np.asarray([float(cue["time"]) for cue in cues])
    cue_frames = np.asarray([int(time_s * fps) for time_s in cue_times_s])
    start_frame = int(round(WARMUP_SEC * fps))

    print(f"Extracting causal rich features for {total_frames - start_frame} frames")
    full_frames = np.arange(start_frame, total_frames, dtype=np.int64)
    full_features = np.asarray(
        [
            extract_features_rich(keypoints, int(frame), fps, total_frames)
            for frame in full_frames
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(full_features)):
        raise RuntimeError("non-finite values in extracted detector features")

    sample_frames, sample_labels, _sample_cue_ids = build_sample_index(
        cue_frames,
        fps=fps,
        start_frame=start_frame,
        end_frame=total_frames,
    )
    sample_features = full_features[sample_frames - start_frame]
    folds = cue_balanced_folds(
        cue_frames,
        start_frame=start_frame,
        end_frame=total_frames,
        n_folds=N_FOLDS,
    )

    crossfit_scores = np.full(total_frames - start_frame, np.nan, dtype=np.float64)
    fold_index = np.full(total_frames - start_frame, -1, dtype=np.int16)
    fold_records: list[dict] = []
    fire_records: list[dict] = []
    purge_frames = int(round(PURGE_SEC * fps))
    feature_names = _all_rich_names()

    for fold_id, (test_start, test_stop) in enumerate(folds):
        train_mask = (
            (sample_frames < test_start - purge_frames)
            | (sample_frames >= test_stop + purge_frames)
        )
        x_train = sample_features[train_mask]
        y_train = sample_labels[train_mask]
        if len(np.unique(y_train)) != 2:
            raise RuntimeError(f"fold {fold_id} training data lost a class")

        selected = select_features(
            x_train,
            y_train,
            n_features=N_SELECTED_FEATURES,
        )
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        )
        classifier.fit(x_train[:, selected], y_train)

        test_slice = slice(test_start - start_frame, test_stop - start_frame)
        scores = classifier.predict_proba(full_features[test_slice][:, selected])[:, 1]
        crossfit_scores[test_slice] = scores
        fold_index[test_slice] = fold_id

        context_start = max(start_frame, test_start - int(round(3.0 * fps)))
        context_slice = slice(context_start - start_frame, test_stop - start_frame)
        context_scores = classifier.predict_proba(
            full_features[context_slice][:, selected]
        )[:, 1]
        local_fires = detect_peaks(context_scores, fps, args.threshold)
        smoothed = causal_smooth(context_scores, 2)
        kept_fires = []
        for local_index in local_fires:
            frame = context_start + int(local_index)
            if test_start <= frame < test_stop:
                kept_fires.append(frame)
                fire_records.append(
                    {
                        "frame": frame,
                        "time_s": frame / fps,
                        "score_at_emission": float(smoothed[local_index]),
                        "fold": fold_id,
                    }
                )

        fold_cues = np.flatnonzero(
            (cue_frames >= test_start) & (cue_frames < test_stop)
        )
        fold_records.append(
            {
                "fold": fold_id,
                "test_start_frame": test_start,
                "test_stop_frame": test_stop,
                "test_start_s": test_start / fps,
                "test_stop_s": test_stop / fps,
                "held_out_cue_indices": fold_cues.tolist(),
                "n_train_positive_frames": int(np.sum(y_train == 1)),
                "n_train_negative_frames": int(np.sum(y_train == 0)),
                "selected_feature_indices": selected.tolist(),
                "selected_feature_names": [feature_names[index] for index in selected],
                "emission_frames": kept_fires,
            }
        )

    if np.any(~np.isfinite(crossfit_scores)) or np.any(fold_index < 0):
        raise RuntimeError("blocked cross-fitting left unscored frames")

    fire_records.sort(key=lambda item: item["frame"])
    fire_times_s = np.asarray([record["time_s"] for record in fire_records])
    metrics = {
        str(tolerance): event_metrics(cue_times_s, fire_times_s, tolerance)
        for tolerance in EVENT_TOLERANCES_S
    }
    boundaries_s = [stop / fps for _, stop in folds[:-1]]
    boundary_nearby = [
        {
            "fire_index": index,
            "time_s": float(time_s),
            "nearest_boundary_distance_s": float(
                min(abs(time_s - boundary) for boundary in boundaries_s)
            ),
        }
        for index, time_s in enumerate(fire_times_s)
        if min(abs(time_s - boundary) for boundary in boundaries_s) <= 1.5
    ]

    payload = {
        "schema_version": 1,
        "status": "visual_blocked_crossfit_detector_complete",
        "scope": "One Firebird performance; five contiguous held-out time blocks",
        "claim_limit": (
            "Within-recording blocked cross-fit only. This does not estimate "
            "performance on a new conductor or production. The 0.60 operating "
            "threshold was frozen before the all-cue transition experiment."
        ),
        "method": {
            "folds": N_FOLDS,
            "fold_construction": "contiguous cue-balanced blocks with boundaries between cues",
            "purge_s": PURGE_SEC,
            "raw_features": int(full_features.shape[1]),
            "selected_features_per_fold": N_SELECTED_FEATURES,
            "feature_selection": "standardized LDA coefficient magnitude, fitted inside each training fold",
            "classifier": "RandomForestClassifier",
            "classifier_parameters": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "n_jobs": 1,
                "random_state": 42,
            },
            "threshold": args.threshold,
            "peak_detector": "causal peak-confirmation state machine from live_detector.py",
            "fps": fps,
        },
        "provenance": {
            "repository_revision_before_run": repository_revision(),
            "script": repository_path(Path(__file__)),
            "annotations": {
                "path": repository_path(args.annotations),
                "sha256": sha256(args.annotations),
            },
            "keypoint_cache": {
                "path": repository_path(args.cache),
                "sha256": sha256(args.cache),
            },
        },
        "n_cues": len(cues),
        "n_emissions": len(fire_records),
        "emissions_per_minute": len(fire_records) / (total_frames / fps) * 60,
        "cues": [
            {
                "cue_index": index,
                "time_s": float(cue["time"]),
                "frame": int(cue_frames[index]),
                "label": cue["label"],
            }
            for index, cue in enumerate(cues)
        ],
        "emissions": [
            {"emission_index": index, **record}
            for index, record in enumerate(fire_records)
        ],
        "event_metrics": metrics,
        "folds": fold_records,
        "fold_boundary_audit": {
            "boundaries_s": boundaries_s,
            "emissions_within_1_5s": boundary_nearby,
        },
    }

    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_result.replace(result_path)
    temporary_trace = trace_path.with_suffix(".npz.tmp")
    with temporary_trace.open("wb") as handle:
        np.savez_compressed(
            handle,
            frame=full_frames,
            time_s=full_frames / fps,
            crossfit_score=crossfit_scores,
            fold=fold_index,
            emission_frame=np.asarray([item["frame"] for item in fire_records]),
            emission_time_s=fire_times_s,
        )
    temporary_trace.replace(trace_path)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {result_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
