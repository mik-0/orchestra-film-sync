"""Run one pinned published online audio tracker against an approved reference."""

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
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


PROJECT_ROOT = _AUDIO_ROOT
REPOSITORY_ROOT = _REPO_ROOT
SCRIPTS = _REPO_ROOT / "shared/analysis"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from analyze_audio_failures import (  # noqa: E402
    centred_rms,
    crop,
    enrich_episodes,
    error_metrics,
    moving_block_spearman_interval,
    repository_revision,
    sha256,
)
from online_dtw.alignment_reference import failure_episodes  # noqa: E402
from online_dtw.audio_io import load_mono  # noqa: E402
from online_dtw.presets import Preset  # noqa: E402


CACHE = PROJECT_ROOT / ".cache-published-baselines"
METHODS = (
    "external_matchmaker_arzt_v0_3_0",
    "external_matchmaker_dixon_v0_3_0",
    "external_concertcue_otw_2876aaca",
)
MATCHMAKER_VERSION = "0.3.0"
MATCHMAKER_COMMIT = "0d106d07d96f9def77de116b29690c262b51b9ee"
CONCERTCUE_COMMIT = "2876aacae008aa75a1512b70e3aca0bca653b4fa"
CONCERTCUE_ARCHIVE_SHA256 = (
    "b41766371e3035ee4ec086f22848cce726b9c569618e95c1ce15dacbfb258bcf"
)
CONCERTCUE_MAX_MATRIX_BYTES = 1024**3


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE)
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


def load_span_direct(path: Path, *, start: float, end: float, sr: int) -> np.ndarray:
    """Decode only the requested span instead of materializing a full-file cache."""
    import librosa

    audio, _ = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        offset=float(start),
        duration=float(end - start),
        dtype=np.float32,
    )
    return audio


def sampled_rms_from_file(
    path: Path,
    sample_t: np.ndarray,
    *,
    window_seconds: float,
) -> np.ndarray:
    """Read only the small source-file windows needed for RMS samples."""
    import soundfile as sf

    values = np.empty(sample_t.shape, dtype=np.float64)
    try:
        handle = sf.SoundFile(path)
    except sf.LibsndfileError:
        # libsndfile does not decode containers such as the Firebird MOV. The
        # fallback is intentionally full-file and used only for that short
        # excerpt; long MP3 sources stay on the sampled seek path below.
        import librosa

        audio, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)
        return centred_rms(
            audio,
            sample_t,
            span_start=0.0,
            sr=int(sr),
            window_seconds=window_seconds,
        )
    with handle:
        sr = int(handle.samplerate)
        window = max(1, int(round(window_seconds * sr)))
        half = window // 2
        for index, seconds in enumerate(sample_t):
            centre = int(round(float(seconds) * sr))
            start = max(0, centre - half)
            handle.seek(start)
            audio = handle.read(window, dtype="float32", always_2d=True)
            if audio.size == 0:
                values[index] = 0.0
            else:
                mono = np.mean(audio, axis=1, dtype=np.float64)
                values[index] = float(np.sqrt(np.mean(np.square(mono))))
    return values


def compute_matchmaker_features_chunked(
    audio: np.ndarray,
    *,
    processor,
    hop: int,
    chunk_frames: int = 600,
) -> np.ndarray:
    """Compute the upstream center=False feature stream without a giant STFT."""
    n_fft = 2 * hop
    frame_count = max(0, 1 + (len(audio) + hop - n_fft) // hop)
    blocks = []
    for first_frame in range(0, frame_count, chunk_frames):
        count = min(chunk_frames, frame_count - first_frame)
        padded_start = first_frame * hop
        padded_end = padded_start + (count - 1) * hop + n_fft
        segment = np.zeros(padded_end - padded_start, dtype=audio.dtype)
        source_start = max(padded_start, hop) - hop
        source_end = min(padded_end, len(audio) + hop) - hop
        if source_end > source_start:
            destination_start = max(padded_start, hop) - padded_start
            segment[destination_start: destination_start + source_end - source_start] = (
                audio[source_start:source_end]
            )
        features, _ = processor((segment, first_frame * hop / 44100))
        blocks.append(features)
    if not blocks:
        return np.empty((0, 75), dtype=np.float32)
    return np.vstack(blocks)


def frame_times(
    count: int,
    *,
    sample_rate: int,
    hop_samples: int,
    fft_samples: int,
    centred_with_left_padding: bool,
) -> dict[str, np.ndarray]:
    """Return window-start, feature-centre and causal-availability times.

    Matchmaker's wrapper pads one half-window on the left, so frame zero is
    centred at live time zero and becomes available one hop later. ConcertCue
    does not pad: frame zero starts at live time zero and becomes available
    after the complete FFT window has arrived.
    """

    index = np.arange(count, dtype=np.float64)
    if centred_with_left_padding:
        centre = index * hop_samples / sample_rate
        start = centre - fft_samples / (2 * sample_rate)
    else:
        start = index * hop_samples / sample_rate
        centre = start + fft_samples / (2 * sample_rate)
    availability = start + fft_samples / sample_rate
    return {
        "window_start": start,
        "feature_center": centre,
        "availability": availability,
    }


def calibration_metrics(
    positions: np.ndarray,
    timing: dict[str, np.ndarray],
    *,
    warmup_s: float = 5.0,
) -> dict:
    """Describe raw identical-input output under each timestamp convention."""

    if any(values.shape != positions.shape for values in timing.values()):
        raise ValueError("calibration positions and timestamp arrays must match")
    keep = timing["availability"] >= warmup_s
    if not np.any(keep):
        raise ValueError("calibration stream is shorter than the warm-up")

    def describe(target: np.ndarray) -> dict:
        error = positions[keep] - target[keep]
        absolute = np.abs(error)
        return {
            "median_signed_error_s": float(np.median(error)),
            "median_abs_error_s": float(np.median(absolute)),
            "p90_abs_error_s": float(np.percentile(absolute, 90)),
            "max_abs_error_s": float(np.max(absolute)),
        }

    return {
        "protocol": "complete live source followed against itself",
        "warmup_s": warmup_s,
        "n_frames_after_warmup": int(np.sum(keep)),
        "raw_position_relative_to_window_start": describe(timing["window_start"]),
        "raw_position_relative_to_feature_center": describe(timing["feature_center"]),
        "raw_position_relative_to_availability": describe(timing["availability"]),
        "interpretation": (
            "The availability-relative error is the algorithmic lag observable "
            "before audio-driver, operating-system, output-device, or film-pipeline latency."
        ),
    }


def make_matchmaker_follower(
    reference_features: np.ndarray,
    *,
    method: str,
    frame_rate: int,
):
    from matchmaker.dp import (  # type: ignore[import-not-found]
        OnlineTimeWarpingArztFrame,
        OnlineTimeWarpingDixonFrame,
    )

    reference_times = np.arange(
        reference_features.shape[0], dtype=np.float32
    ) / frame_rate
    if method == "external_matchmaker_arzt_v0_3_0":
        follower = OnlineTimeWarpingArztFrame(
            reference_features=reference_features,
            score_positions=reference_times,
            ref_frame_to_beat=reference_times,
            frame_rate=frame_rate,
        )
        decoder_name = "OnlineTimeWarpingArztFrame"
        decoder_config = {
            "window_size_s": 10,
            "step_size": 3,
            "start_window_size_s": 0.1,
            "distance": "Manhattan",
        }
    else:
        follower = OnlineTimeWarpingDixonFrame(
            reference_features=reference_features,
            score_positions=reference_times,
            ref_frame_to_beat=reference_times,
            frame_rate=frame_rate,
        )
        decoder_name = "OnlineTimeWarpingDixonFrame"
        decoder_config = {
            "window_size_s": 10,
            "max_run_count": 3,
            "distance": "Euclidean",
        }
    return follower, decoder_name, decoder_config


def decode_matchmaker(
    reference_features: np.ndarray,
    live_features: np.ndarray,
    *,
    method: str,
    frame_rate: int,
    progress: bool,
) -> tuple[np.ndarray, dict, dict, float]:
    follower, decoder_name, decoder_config = make_matchmaker_follower(
        reference_features,
        method=method,
        frame_rate=frame_rate,
    )
    positions = np.empty(live_features.shape[0], dtype=np.float64)
    started = time.perf_counter()
    reached_reference_end_frame = None
    for index, observation in enumerate(live_features):
        if not follower.is_still_following():
            reached_reference_end_frame = index
            positions[index:] = positions[index - 1] if index else 0.0
            break
        positions[index] = follower(observation, index / frame_rate)
        if progress and index > 0 and index % 10000 == 0:
            print(f"decoded {index}/{len(live_features)} frames", flush=True)
    runtime = time.perf_counter() - started
    upstream_latency = dict(getattr(follower, "latency_stats", {}))
    if hasattr(follower, "input_features"):
        follower.input_features.clear()
    if hasattr(follower, "_alignment_path"):
        follower._alignment_path.clear()
    decoder = {
        "name": decoder_name,
        "config": decoder_config,
        "reached_reference_end_frame": reached_reference_end_frame,
        "post_end_behavior": "hold_last_published_position",
        "mean_batch_compute_s_per_frame": runtime / max(1, live_features.shape[0]),
        "upstream_latency_stats": upstream_latency,
    }
    return positions, decoder, decoder_config, runtime


def run_matchmaker(
    live_path: Path,
    ref_path: Path,
    *,
    method: str,
    span: dict,
) -> dict:
    """Return raw positions with centre and causal-availability timestamps."""
    version = importlib.metadata.version("pymatchmaker")
    if version != MATCHMAKER_VERSION:
        raise RuntimeError(
            f"expected pymatchmaker {MATCHMAKER_VERSION}, found {version}"
        )
    from matchmaker.features.audio import (  # type: ignore[import-not-found]
        LogSpectralEnergyProcessor,
    )

    sr = 44100
    hop = 1470
    frame_rate = 30
    feature_started = time.perf_counter()
    ref_audio = load_span_direct(
        ref_path,
        start=float(span["ref_start"]),
        end=float(span["ref_end"]),
        sr=sr,
    )
    print(f"decoded reference: {len(ref_audio) / sr:.1f}s", flush=True)
    reference_features = compute_matchmaker_features_chunked(
        ref_audio,
        processor=LogSpectralEnergyProcessor(
            sample_rate=sr,
            hop_length=hop,
        ),
        hop=hop,
    )
    print(f"reference features: {reference_features.shape}", flush=True)
    del ref_audio
    live_audio = load_span_direct(
        live_path,
        start=float(span["live_start"]),
        end=float(span["live_end"]),
        sr=sr,
    )
    print(f"decoded live: {len(live_audio) / sr:.1f}s", flush=True)
    live_features = compute_matchmaker_features_chunked(
        live_audio,
        processor=LogSpectralEnergyProcessor(
            sample_rate=sr,
            hop_length=hop,
        ),
        hop=hop,
    )
    print(f"live features: {live_features.shape}", flush=True)
    del live_audio
    feature_runtime = time.perf_counter() - feature_started
    print(f"running published {method}", flush=True)
    positions, decoder, decoder_config, decoder_runtime = decode_matchmaker(
        reference_features,
        live_features,
        method=method,
        frame_rate=frame_rate,
        progress=True,
    )
    print("calibrating published method on identical live input", flush=True)
    calibration_positions, _, _, calibration_runtime = decode_matchmaker(
        live_features,
        live_features,
        method=method,
        frame_rate=frame_rate,
        progress=False,
    )
    timing = frame_times(
        live_features.shape[0],
        sample_rate=sr,
        hop_samples=hop,
        fft_samples=2 * hop,
        centred_with_left_padding=True,
    )
    calibration = calibration_metrics(calibration_positions, timing)
    calibration["decoder_runtime_s"] = calibration_runtime
    config = {
        "implementation": "pymatchmaker",
        "version": version,
        "release_commit": MATCHMAKER_COMMIT,
        "decoder": decoder["name"],
        "feature": "log_spectral",
        "sample_rate_hz": sr,
        "hop_samples": hop,
        "fft_samples": 2 * hop,
        "frame_rate_hz": frame_rate,
        "decoder_config": decoder_config,
        "reached_reference_end_frame": decoder["reached_reference_end_frame"],
        "post_end_behavior": decoder["post_end_behavior"],
        "feature_timing": {
            "reference_position_convention": "feature_center",
            "live_feature_timestamp_convention": "feature_center",
            "evaluation_timestamp_convention": "complete_window_availability",
            "window_start_to_center_s": hop / sr,
            "center_to_availability_s": hop / sr,
            "window_start_to_availability_s": 2 * hop / sr,
        },
        "decoder_compute": {
            "mean_batch_s_per_frame": decoder["mean_batch_compute_s_per_frame"],
            "upstream_latency_stats": decoder["upstream_latency_stats"],
        },
    }
    return {
        "window_start_t": timing["window_start"],
        "feature_center_t": timing["feature_center"],
        "availability_t": timing["availability"],
        "reference_position_t": positions,
        "tracker": config,
        "timing_calibration": calibration,
        "feature_runtime_s": feature_runtime,
        "decoder_runtime_s": decoder_runtime,
    }


def concertcue_matrix_bytes(reference_samples: int) -> tuple[int, int]:
    n_fft = 8192
    ref_hop = 4096
    frames = max(0, (reference_samples - n_fft) // ref_hop + 1)
    distance_bytes = frames * (frames * 4) * np.dtype(np.float32).itemsize
    live_bytes = 12 * (frames * 4) * np.dtype(np.float32).itemsize
    return frames, int(distance_bytes + live_bytes)


def run_concertcue(
    live_audio: np.ndarray,
    ref_audio: np.ndarray,
    *,
    cache: Path,
) -> dict:
    archive = cache / "web-score-following/web/public/python/libaudio_rt.zip"
    if not archive.exists():
        raise RuntimeError(
            f"ConcertCue source is missing: run setup_published_baselines.py ({archive})"
        )
    if sha256(archive) != CONCERTCUE_ARCHIVE_SHA256:
        raise RuntimeError("ConcertCue tracker archive hash does not match the protocol")

    ref_frames, allocation_bytes = concertcue_matrix_bytes(len(ref_audio))
    if allocation_bytes > CONCERTCUE_MAX_MATRIX_BYTES:
        raise RuntimeError(
            "exact ConcertCue implementation excluded by frozen 1 GiB allocation rule: "
            f"{ref_frames} reference frames require approximately "
            f"{allocation_bytes / 1024**3:.2f} GiB"
        )

    dummy_js = types.ModuleType("js")
    dummy_js.Blob = None
    dummy_js.document = None
    dummy_js.window = None
    sys.modules.setdefault("js", dummy_js)
    sys.path.insert(0, str(archive))
    import libaudio_rt as concertcue  # type: ignore[import-not-found]

    sr = 44100
    n_fft = 8192
    ref_hop = 4096
    live_hop = 3686
    params = {"c": 300, "max_run_count": 3, "diag_weight": 0.4}
    feature_started = time.perf_counter()
    reference_features = concertcue.audio_to_np_cens(
        ref_audio,
        sr,
        n_fft,
        ref_hop,
    )
    chroma_maker = concertcue.ChromaMaker(sr, n_fft)
    live_frames = max(0, (len(live_audio) - n_fft) // live_hop + 1)
    live_features = np.empty((12, live_frames), dtype=np.float64)
    for index in range(live_frames):
        start = index * live_hop
        live_features[:, index] = chroma_maker.insert(
            live_audio[start: start + n_fft]
        )
    feature_runtime = time.perf_counter() - feature_started

    follower = concertcue.OTW(reference_features, params)
    positions = np.empty(live_frames, dtype=np.float64)
    decoder_started = time.perf_counter()
    for index in range(live_frames):
        positions[index] = follower.insert(live_features[:, index]) * ref_hop / sr
    decoder_runtime = time.perf_counter() - decoder_started

    calibration_started = time.perf_counter()
    calibration_reference = concertcue.audio_to_np_cens(
        live_audio,
        sr,
        n_fft,
        ref_hop,
    )
    calibration_follower = concertcue.OTW(calibration_reference, params)
    calibration_positions = np.empty(live_frames, dtype=np.float64)
    for index in range(live_frames):
        calibration_positions[index] = (
            calibration_follower.insert(live_features[:, index]) * ref_hop / sr
        )
    calibration_runtime = time.perf_counter() - calibration_started
    timing = frame_times(
        live_frames,
        sample_rate=sr,
        hop_samples=live_hop,
        fft_samples=n_fft,
        centred_with_left_padding=False,
    )
    calibration = calibration_metrics(calibration_positions, timing)
    calibration["decoder_and_reference_feature_runtime_s"] = calibration_runtime
    config = {
        "implementation": "ConcertCue libaudio_rt 1.0",
        "commit": CONCERTCUE_COMMIT,
        "tracker_archive_sha256": CONCERTCUE_ARCHIVE_SHA256,
        "decoder": "OTW",
        "feature": "upstream 12-bin CENS-like chroma",
        "sample_rate_hz": sr,
        "reference_hop_samples": ref_hop,
        "live_hop_samples": live_hop,
        "fft_samples": n_fft,
        "decoder_config": params,
        "estimated_preallocation_bytes": allocation_bytes,
        "feature_timing": {
            "reference_position_convention": "window_start",
            "live_feature_timestamp_convention": "window_start_in_upstream_wrapper",
            "evaluation_timestamp_convention": "complete_window_availability",
            "window_start_to_center_s": n_fft / (2 * sr),
            "center_to_availability_s": n_fft / (2 * sr),
            "window_start_to_availability_s": n_fft / sr,
        },
        "decoder_compute": {
            "mean_batch_s_per_frame": decoder_runtime / max(1, live_frames),
        },
    }
    return {
        "window_start_t": timing["window_start"],
        "feature_center_t": timing["feature_center"],
        "availability_t": timing["availability"],
        "reference_position_t": positions,
        "tracker": config,
        "timing_calibration": calibration,
        "feature_runtime_s": feature_runtime,
        "decoder_runtime_s": decoder_runtime,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    trace_output = args.trace_output or args.output.with_name(
        args.output.stem + "_trace.npz"
    )
    for path in (args.output, trace_output):
        if path.exists() and not args.force:
            raise SystemExit(f"refusing to replace existing output without --force: {path}")

    preset = Preset.load(args.preset)
    live_path = args.live or preset.audio.live
    ref_path = args.ref or preset.audio.ref
    with args.reference.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    if not reference.get("structural_qc", {}).get("passed", False):
        raise SystemExit("reference did not pass structural QC")
    span = reference["shared_span"]
    dense_path = args.reference.parent / reference["dense_path"]
    with np.load(dense_path) as dense:
        dense_live_t = dense["live_t"].astype(np.float64)
        dense_ref_t = dense["ref_t"].astype(np.float64)

    print(f"pair: {preset.name}", flush=True)
    print(f"method: {args.method}", flush=True)
    if args.method.startswith("external_matchmaker_"):
        print(
            f"loading and featurizing {ref_path.name}, then {live_path.name}, at 44100 Hz",
            flush=True,
        )
        result = run_matchmaker(
            live_path,
            ref_path,
            method=args.method,
            span=span,
        )
    else:
        print(f"loading {live_path.name} and {ref_path.name} at 44100 Hz", flush=True)
        live_full, _ = load_mono(live_path, target_sr=44100)
        ref_full, _ = load_mono(ref_path, target_sr=44100)
        live_audio = crop(live_full, span["live_start"], span["live_end"], 44100)
        ref_audio = crop(ref_full, span["ref_start"], span["ref_end"], 44100)
        result = run_concertcue(live_audio, ref_audio, cache=args.cache.resolve())
    availability_live_t = result["availability_t"] + float(span["live_start"])
    feature_center_live_t = result["feature_center_t"] + float(span["live_start"])
    window_start_live_t = result["window_start_t"] + float(span["live_start"])
    online_ref_t = result["reference_position_t"] + float(span["ref_start"])
    online_live_t = availability_live_t
    inside = (online_live_t >= dense_live_t[0]) & (online_live_t <= dense_live_t[-1])
    online_live_t = online_live_t[inside]
    feature_center_live_t = feature_center_live_t[inside]
    window_start_live_t = window_start_live_t[inside]
    online_ref_t = online_ref_t[inside]
    if online_live_t.size == 0:
        raise RuntimeError("tracker emitted no positions inside the approved span")
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
        np.ceil(
            max(float(dense_live_t[0]), float(online_live_t[0]))
            / args.sample_step
        ) * args.sample_step,
        min(float(dense_live_t[-1]), float(online_live_t[-1])) + 1e-9,
        args.sample_step,
    )
    sample_error = np.interp(sample_t, online_live_t, error_s)
    if args.method.startswith("external_matchmaker_"):
        rms = sampled_rms_from_file(
            live_path,
            sample_t,
            window_seconds=args.energy_window,
        )
    else:
        rms = centred_rms(
            live_audio,
            sample_t,
            span_start=float(span["live_start"]),
            sr=44100,
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
    evaluated_duration = float(online_live_t[-1] - online_live_t[0])
    payload = {
        "pair": preset.name,
        "method": args.method,
        "status": "provisional_until_conflict_review_is_complete",
        "reference": {
            "path": repository_path(args.reference),
            "sha256": sha256(args.reference),
            "dense_path": repository_path(dense_path),
            "dense_sha256": sha256(dense_path),
            "shared_span": span,
        },
        "tracker": result["tracker"],
        "timing_calibration": result["timing_calibration"],
        "causality": {
            "evaluation_time_basis": "complete_live_analysis_window_availability",
            "future_live_audio_relative_to_evaluation_time_used": False,
            "future_live_audio_relative_to_feature_center_used": True,
            "complete_prerecorded_reference_available": True,
            "evaluation_reference_available_to_tracker": False,
        },
        "failure_definition": {
            "absolute_error_threshold_s": args.threshold,
            "minimum_duration_s": args.minimum_duration,
        },
        "metrics": metrics,
        "failure_summary": {
            "episode_count": len(episodes),
            "total_failure_duration_s": failing_duration,
            "failure_time_fraction": failing_duration / max(
                evaluated_duration, np.finfo(float).eps
            ),
            "episodes_starting_after_low_energy": int(
                sum(item["starts_after_low_energy"] for item in episodes)
            ),
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
            "feature_runtime_seconds": result["feature_runtime_s"],
            "decoder_runtime_seconds": result["decoder_runtime_s"],
            "total_runtime_seconds": (
                result["feature_runtime_s"] + result["decoder_runtime_s"]
            ),
            "live_file": live_path.name,
            "ref_file": ref_path.name,
            "trace_path": repository_path(trace_output),
        },
    }
    write_json(args.output, payload)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_trace = trace_output.with_suffix(trace_output.suffix + ".tmp")
    with temporary_trace.open("wb") as handle:
        np.savez_compressed(
            handle,
            online_live_t=online_live_t,
            availability_live_t=online_live_t,
            feature_center_live_t=feature_center_live_t,
            window_start_live_t=window_start_live_t,
            online_ref_t=online_ref_t,
            ground_truth_ref_t=ground_truth_ref_t,
            error_s=error_s,
            confidence=np.full(online_live_t.shape, np.nan),
            sample_t=sample_t,
            sample_error_s=sample_error,
            sample_energy_ratio=energy_ratio,
            sample_confidence=np.full(sample_t.shape, np.nan),
        )
    temporary_trace.replace(trace_output)
    print(
        f"median={metrics['median_abs_error_s']:.3f}s "
        f"p90={metrics['p90_abs_error_s']:.3f}s "
        f"within1s={metrics['rate_within_1s']:.1%} "
        f"failure_fraction={payload['failure_summary']['failure_time_fraction']:.1%}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {trace_output}", flush=True)


if __name__ == "__main__":
    main()
