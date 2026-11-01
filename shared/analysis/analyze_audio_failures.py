"""Run one preset and map its tracker failures against an audited dense path.

The dense Sync Toolbox path supplies the descriptive reference trajectory.  A
separate one-second grid is used for condition summaries, while contiguous
frame-level errors define failure episodes.  Adjacent seconds are not treated
as independent replicates: the error-energy association is reported per
recording with a moving-block bootstrap interval and no naive p-value.
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
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import warnings

HERE = _AUDIO_ROOT
sys.path.insert(0, str(HERE))

from online_dtw.rank_bootstrap import moving_block_rank_correlation
from online_dtw.alignment_reference import failure_episodes
from online_dtw.audio_io import load_mono
from online_dtw.presets import Preset
from evaluate_anchors import run_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--backend", choices=("hmm", "oltw"))
    parser.add_argument("--method-label")
    parser.add_argument("--reanchor-interval", type=float, default=0.0)
    parser.add_argument("--reanchor-live-window", type=float, default=30.0)
    parser.add_argument("--reanchor-ref-search", type=float, default=60.0)
    parser.add_argument("--reanchor-snap-threshold", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--minimum-duration", type=float, default=1.0)
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--energy-window", type=float, default=0.37)
    parser.add_argument("--low-energy-ratio", type=float, default=0.10)
    parser.add_argument("--bootstrap-block", type=int, default=30)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def crop(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    return audio[int(round(start * sr)): int(round(end * sr))]


def centred_rms(
    audio: np.ndarray,
    sample_t: np.ndarray,
    *,
    span_start: float,
    sr: int,
    window_seconds: float,
) -> np.ndarray:
    half = max(1, int(round(window_seconds * sr / 2)))
    squared = np.square(audio.astype(np.float64))
    cumulative = np.r_[0.0, np.cumsum(squared)]
    centres = np.round((sample_t - span_start) * sr).astype(np.int64)
    lo = np.clip(centres - half, 0, len(audio))
    hi = np.clip(centres + half, 0, len(audio))
    counts = np.maximum(1, hi - lo)
    return np.sqrt((cumulative[hi] - cumulative[lo]) / counts)


def moving_block_spearman_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    # Preserve this caller's historical pairwise finite-value filtering. The
    # cached-trace reanalysis validates complete regular grids before calling
    # the shared resampler, so it never joins observations across missing times.
    mask = np.isfinite(x) & np.isfinite(y)
    result = moving_block_rank_correlation(
        x[mask], y[mask], block_size=block_size,
        repetitions=repetitions, seed=seed,
    )
    if result.undefined_resamples:
        warnings.warn(
            f"{result.undefined_resamples}/{repetitions} bootstrap correlations "
            "were undefined; bounds use only defined resamples",
            RuntimeWarning,
            stacklevel=2,
        )
    return tuple(
        float(value) if value is not None else float("nan")
        for value in (result.rho, result.lower, result.upper)
    )


def error_metrics(error: np.ndarray, *, threshold: float) -> dict:
    absolute = np.abs(error[np.isfinite(error)])
    return {
        "n_frames": int(absolute.size),
        "median_abs_error_s": float(np.median(absolute)),
        "mean_abs_error_s": float(np.mean(absolute)),
        "p90_abs_error_s": float(np.percentile(absolute, 90)),
        "max_abs_error_s": float(np.max(absolute)),
        "median_signed_error_s": float(np.median(error[np.isfinite(error)])),
        "rate_within_250ms": float(np.mean(absolute <= 0.25)),
        "rate_within_500ms": float(np.mean(absolute <= 0.50)),
        "rate_within_1s": float(np.mean(absolute <= 1.0)),
        "rate_beyond_failure_threshold": float(np.mean(absolute >= threshold)),
    }


def enrich_episodes(
    episodes: list[dict],
    *,
    sample_t: np.ndarray,
    energy_ratio: np.ndarray,
    low_energy_ratio: float,
) -> list[dict]:
    enriched = []
    for episode in episodes:
        start = float(episode["live_start"])
        end = float(episode["live_end"])
        inside = (sample_t >= start) & (sample_t <= end)
        before = (sample_t >= start - 1.0) & (sample_t < start)
        item = dict(episode)
        item["median_energy_ratio"] = (
            float(np.median(energy_ratio[inside])) if np.any(inside) else None
        )
        item["pre_1s_energy_ratio"] = (
            float(np.mean(energy_ratio[before])) if np.any(before) else None
        )
        item["starts_after_low_energy"] = bool(
            np.any(before)
            and float(np.mean(energy_ratio[before])) <= low_energy_ratio
        )
        enriched.append(item)
    return enriched


def main() -> None:
    args = parse_args()
    trace_output = args.trace_output or args.output.with_name(
        args.output.stem + "_trace.npz"
    )
    for path in (args.output, trace_output):
        if path.exists() and not args.force:
            raise SystemExit(f"refusing to replace existing output without --force: {path}")

    preset = Preset.load(args.preset)
    backend = args.backend or preset.stream.backend
    if args.reanchor_interval > 0 and backend != "hmm":
        raise SystemExit("periodic re-anchoring is only implemented for HMM")
    live_path = args.live or preset.audio.live
    ref_path = args.ref or preset.audio.ref
    with open(args.reference, encoding="utf-8") as handle:
        reference = json.load(handle)
    if not reference.get("structural_qc", {}).get("passed", False):
        raise SystemExit("reference did not pass structural QC")
    span = reference["shared_span"]
    dense_path = args.reference.parent / reference["dense_path"]
    dense = np.load(dense_path)
    dense_live_t = dense["live_t"]
    dense_ref_t = dense["ref_t"]

    print(f"pair: {preset.name}")
    print(f"reference: {args.reference}")
    live_full, _ = load_mono(live_path, target_sr=preset.feature.sr)
    ref_full, _ = load_mono(ref_path, target_sr=preset.feature.sr)
    live_audio = crop(
        live_full, span["live_start"], span["live_end"], preset.feature.sr
    )
    ref_audio = crop(
        ref_full, span["ref_start"], span["ref_end"], preset.feature.sr
    )
    hop_sec = preset.feature.hop / preset.feature.sr
    start_ref_relative = float(dense_ref_t[0]) - float(span["ref_start"])
    start_j = max(0, int(round(start_ref_relative / hop_sec)))
    print(
        f"running {backend} with {preset.feature.name}; "
        f"live={len(live_audio)/preset.feature.sr:.1f}s "
        f"ref={len(ref_audio)/preset.feature.sr:.1f}s"
    )
    if backend == "hmm":
        decoder_overrides = {
            "max_advance": preset.hmm.max_advance,
            "search_width": preset.hmm.search_width,
            "backward_penalty": preset.hmm.backward_penalty,
        }
    else:
        decoder_overrides = asdict(preset.oltw)
        # ``silence_rms`` belongs to the streaming wrapper, not the
        # frame-level OLTW decoder used by this offline simulation.
        decoder_overrides.pop("silence_rms", None)
    started = time.perf_counter()
    result = run_tracker(
        live_audio=live_audio,
        ref_audio=ref_audio,
        sr=preset.feature.sr,
        hop=preset.feature.hop,
        feature=preset.feature.name,
        start_j=start_j,
        backend=backend,
        beta=preset.hmm.beta,
        skip_mass=preset.hmm.skip_mass,
        emission_buffer_frames=preset.hmm.emission_buffer_frames,
        offline_features=True,
        reanchor_interval=args.reanchor_interval,
        reanchor_live_window=args.reanchor_live_window,
        reanchor_ref_search=args.reanchor_ref_search,
        reanchor_snap_threshold=args.reanchor_snap_threshold,
        oltw_overrides=decoder_overrides,
    )
    runtime = time.perf_counter() - started
    online_live_t = result["live_time"] + float(span["live_start"])
    online_ref_t = result["ref_time"] + float(span["ref_start"])
    inside = (
        (online_live_t >= dense_live_t[0])
        & (online_live_t <= dense_live_t[-1])
    )
    online_live_t = online_live_t[inside]
    online_ref_t = online_ref_t[inside]
    confidence = result["confidence"][inside]
    ground_truth_ref_t = np.interp(online_live_t, dense_live_t, dense_ref_t)
    error_s = online_ref_t - ground_truth_ref_t
    metrics = error_metrics(error_s, threshold=args.threshold)

    episodes = failure_episodes(
        online_live_t,
        error_s,
        threshold_s=args.threshold,
        minimum_duration_s=args.minimum_duration,
    )
    sample_t = np.arange(
        np.ceil(float(dense_live_t[0]) / args.sample_step) * args.sample_step,
        float(dense_live_t[-1]) + 1e-9,
        args.sample_step,
    )
    sample_error = np.interp(sample_t, online_live_t, error_s)
    sample_confidence = np.interp(sample_t, online_live_t, confidence)
    rms = centred_rms(
        live_audio,
        sample_t,
        span_start=float(span["live_start"]),
        sr=preset.feature.sr,
        window_seconds=args.energy_window,
    )
    positive_rms = rms[rms > 0]
    rms_median = float(np.median(positive_rms)) if positive_rms.size else 1.0
    energy_ratio = rms / max(rms_median, np.finfo(float).eps)
    rho, rho_lo, rho_hi = moving_block_spearman_interval(
        np.abs(sample_error),
        energy_ratio,
        block_size=max(1, int(round(args.bootstrap_block / args.sample_step))),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    episodes = enrich_episodes(
        episodes,
        sample_t=sample_t,
        energy_ratio=energy_ratio,
        low_energy_ratio=args.low_energy_ratio,
    )

    failing_duration = float(sum(item["duration_s"] for item in episodes))
    payload = {
        "pair": preset.name,
        "method": args.method_label or (
            "hmm_recovery" if args.reanchor_interval > 0 else backend
        ),
        "status": "provisional_until_conflict_review_is_complete",
        "reference": {
            "path": str(args.reference),
            "sha256": sha256(args.reference),
            "dense_path": str(dense_path),
            "dense_sha256": sha256(dense_path),
            "shared_span": span,
        },
        "tracker": {
            "backend": backend,
            "feature": preset.feature.name,
            "sr": preset.feature.sr,
            "hop": preset.feature.hop,
            "effective_decoder_config": (
                {
                    "search_width": preset.hmm.search_width,
                    "beta": preset.hmm.beta,
                    "sigma_floor": preset.hmm.sigma_floor,
                    "sigma_scale": preset.hmm.sigma_scale,
                    "max_advance": preset.hmm.max_advance,
                    "min_advance": preset.hmm.min_advance,
                    "backward_penalty": preset.hmm.backward_penalty,
                    "skip_mass": preset.hmm.skip_mass,
                    "emission_buffer_frames": (
                        preset.hmm.emission_buffer_frames
                    ),
                }
                if backend == "hmm" else asdict(result["oltw_cfg"])
            ),
            "recovery": {
                "enabled": args.reanchor_interval > 0,
                "interval_s": args.reanchor_interval,
                "live_window_s": args.reanchor_live_window,
                "reference_search_radius_s": args.reanchor_ref_search,
                "snap_threshold_s": args.reanchor_snap_threshold,
                "snap_count": len(result["reanchors"]),
                "snaps": [
                    {
                        "live_t": float(live_t),
                        "old_ref_t": float(old_ref_t),
                        "new_ref_t": float(new_ref_t),
                    }
                    for live_t, old_ref_t, new_ref_t in result["reanchors"]
                ],
            },
            "offline_feature_extraction": True,
        },
        "failure_definition": {
            "absolute_error_threshold_s": args.threshold,
            "minimum_duration_s": args.minimum_duration,
        },
        "metrics": metrics,
        "failure_summary": {
            "episode_count": len(episodes),
            "total_failure_duration_s": failing_duration,
            "failure_time_fraction": failing_duration / (
                float(dense_live_t[-1]) - float(dense_live_t[0])
            ),
            "episodes_starting_after_low_energy": int(sum(
                item["starts_after_low_energy"] for item in episodes
            )),
        },
        "energy_association": {
            "sample_step_s": args.sample_step,
            "energy_window_s": args.energy_window,
            "n_samples": int(sample_t.size),
            "spearman_rho": rho,
            "moving_block_bootstrap_95_interval": [rho_lo, rho_hi],
            "block_seconds": args.bootstrap_block,
            "repetitions": args.bootstrap_repetitions,
            "naive_p_value_reported": False,
        },
        "episodes": episodes,
        "provenance": {
            "repository_revision": repository_revision(),
            "runtime_seconds": runtime,
            "live_file": live_path.name,
            "ref_file": ref_path.name,
            "trace_path": str(trace_output),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_json.replace(args.output)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_trace = trace_output.with_suffix(trace_output.suffix + ".tmp")
    with open(temporary_trace, "wb") as handle:
        np.savez_compressed(
            handle,
            online_live_t=online_live_t,
            online_ref_t=online_ref_t,
            ground_truth_ref_t=ground_truth_ref_t,
            error_s=error_s,
            confidence=confidence,
            sample_t=sample_t,
            sample_error_s=sample_error,
            sample_energy_ratio=energy_ratio,
            sample_confidence=sample_confidence,
        )
    temporary_trace.replace(trace_output)
    print(
        f"episodes={len(episodes)} median={metrics['median_abs_error_s']:.3f}s "
        f"p90={metrics['p90_abs_error_s']:.3f}s max={metrics['max_abs_error_s']:.3f}s"
    )
    print(f"rho(|error|, RMS)={rho:.3f} [{rho_lo:.3f}, {rho_hi:.3f}]")
    print(f"wrote {args.output}")
    print(f"wrote {trace_output}")


if __name__ == "__main__":
    main()
