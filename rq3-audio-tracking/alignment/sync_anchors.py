"""Generate anchor candidates with Sync Toolbox.

Runs the published MrMsDTW audio-to-audio pipeline with quantized chroma and
DLNCO onset features, then samples an ``annotate.py``-compatible anchor JSON.
Pass ``--validate`` with hand-marked anchors to measure the automatic path.

Example::

    THESIS_AUDIO_DIR=/path/to/audio .venv/bin/python rq3-audio-tracking/alignment/sync_anchors.py \
        --preset yuja_fedorova \
        --output output/yuja_fedorova_sync_review.json \
        --n 40 \
        --validate output/anchors.json
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
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np

HERE = _AUDIO_ROOT
sys.path.insert(0, str(HERE))

from online_dtw.audio_io import load_mono
from online_dtw.alignment_reference import sample_mapping, structural_qc
from online_dtw.presets import Preset

try:
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
    from synctoolbox.dtw.utils import (
        compute_optimal_chroma_shift,
        make_path_strictly_monotonic,
        shift_chroma_vectors,
    )
    from synctoolbox.feature.chroma import (
        pitch_to_chroma,
        quantize_chroma,
        quantized_chroma_to_CENS,
    )
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch import audio_to_pitch_features
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    from synctoolbox.feature.utils import estimate_tuning
except ImportError as exc:
    raise SystemExit(
        "Sync Toolbox is not installed. Create the project environment with "
        "`python3 -m venv .venv`, then install the project requirements and "
        "`synctoolbox`."
    ) from exc


FS = 22050
FEATURE_RATE = 50
STEP_WEIGHTS = np.array([1.5, 1.5, 2.0])
THRESHOLD_REC = 10**6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--preset", default="yuja_fedorova")
    parser.add_argument("--live", type=Path, help="override the preset live recording")
    parser.add_argument("--ref", type=Path, help="override the preset reference recording")
    parser.add_argument("--output", type=Path,
                        help="default: output/PRESET_sync_anchors.json")
    parser.add_argument("--n", type=int, default=40,
                        help="number of evenly spaced automatic candidates")
    parser.add_argument("--validate", type=Path,
                        help="compare the dense path with hand-marked anchors")
    parser.add_argument("--live-start", type=float, default=0.0,
                        help="shared musical span start in the live recording")
    parser.add_argument("--live-end", "--max-live-seconds", type=float,
                        help="shared musical span end in the live recording")
    parser.add_argument("--ref-start", type=float, default=0.0,
                        help="shared musical span start in the reference recording")
    parser.add_argument("--ref-end", "--max-ref-seconds", type=float,
                        help="shared musical span end in the reference recording")
    parser.add_argument("--dense-output", type=Path,
                        help="compressed dense path; defaults beside --output")
    parser.add_argument("--no-dense", action="store_true",
                        help="do not write the compressed dense reference path")
    parser.add_argument("--qc-window", type=float, default=10.0,
                        help="seconds per structural path QC interval")
    parser.add_argument("--qc-min-slope", type=float, default=0.25,
                        help="flag a structural stall below this tempo ratio")
    parser.add_argument("--qc-max-slope", type=float, default=4.0,
                        help="flag a structural jump above this tempo ratio")
    parser.add_argument("--allow-qc-fail", action="store_true",
                        help="write output despite structural QC failures")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output file")
    return parser.parse_args()


def extract_features(audio: np.ndarray, tuning_offset: int) -> tuple[np.ndarray, np.ndarray]:
    pitch = audio_to_pitch_features(
        f_audio=audio,
        Fs=FS,
        tuning_offset=tuning_offset,
        feature_rate=FEATURE_RATE,
        verbose=False,
    )
    chroma = quantize_chroma(pitch_to_chroma(f_pitch=pitch))
    pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio,
        Fs=FS,
        tuning_offset=tuning_offset,
        verbose=False,
    )
    dlnco = pitch_onset_features_to_DLNCO(
        f_peaks=pitch_onset,
        feature_rate=FEATURE_RATE,
        feature_sequence_length=chroma.shape[1],
        visualize=False,
    )
    return chroma, dlnco


def sample_candidates(
    live_path: np.ndarray,
    ref_path: np.ndarray,
    count: int,
) -> list[dict]:
    target_live, target_ref = sample_mapping(live_path, ref_path, count)
    return [
        {
            "live_t": round(float(live_t), 4),
            "ref_t": round(float(ref_t), 4),
        }
        for live_t, ref_t in zip(target_live, target_ref)
    ]


def validate(
    live_path: np.ndarray,
    ref_path: np.ndarray,
    anchors_path: Path,
) -> dict | None:
    with open(anchors_path, encoding="utf-8") as handle:
        anchors = json.load(handle)["anchors"]
    hand_live = np.array([float(anchor["live_t"]) for anchor in anchors])
    hand_ref = np.array([float(anchor["ref_t"]) for anchor in anchors])
    keep = (hand_live >= live_path[0]) & (hand_live <= live_path[-1])
    predicted_ref = np.interp(hand_live[keep], live_path, ref_path)
    absolute_error = np.abs(predicted_ref - hand_ref[keep])
    if absolute_error.size == 0:
        print("validation: no hand anchors fall inside the generated path")
        return None
    print("validation against hand anchors:")
    print(f"  n:              {absolute_error.size}")
    print(f"  median |error|: {np.median(absolute_error):.3f} s")
    print(f"  p90 |error|:    {np.percentile(absolute_error, 90):.3f} s")
    print(f"  max |error|:    {np.max(absolute_error):.3f} s")
    print(f"  within 0.25 s:  {100*np.mean(absolute_error <= 0.25):.1f} %")
    print(f"  within 0.50 s:  {100*np.mean(absolute_error <= 0.50):.1f} %")
    print(f"  within 1.00 s:  {100*np.mean(absolute_error <= 1.00):.1f} %")
    return {
        "anchor_file": str(anchors_path),
        "n": int(absolute_error.size),
        "median_abs_error_s": float(np.median(absolute_error)),
        "p90_abs_error_s": float(np.percentile(absolute_error, 90)),
        "max_abs_error_s": float(np.max(absolute_error)),
        "rate_250ms": float(np.mean(absolute_error <= 0.25)),
        "rate_500ms": float(np.mean(absolute_error <= 0.50)),
        "rate_1000ms": float(np.mean(absolute_error <= 1.00)),
    }


def crop_span(
    audio: np.ndarray,
    *,
    start: float,
    end: float | None,
    label: str,
) -> tuple[np.ndarray, float, float]:
    duration = len(audio) / FS
    actual_end = duration if end is None else end
    if start < 0 or actual_end <= start or actual_end > duration + 1 / FS:
        raise SystemExit(
            f"invalid {label} span [{start:.3f}, {actual_end:.3f}] "
            f"for {duration:.3f} s recording"
        )
    i0 = int(round(start * FS))
    i1 = min(len(audio), int(round(actual_end * FS)))
    return audio[i0:i1], start, i1 / FS


def write_dense_path(
    output_path: Path,
    *,
    live_t: np.ndarray,
    ref_t: np.ndarray,
    feature_rate_hz: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            live_t=live_t.astype(np.float64),
            ref_t=ref_t.astype(np.float64),
            feature_rate_hz=np.array(feature_rate_hz, dtype=np.int32),
        )
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    preset = Preset.load(args.preset)
    output_path = args.output or HERE / "output" / f"{preset.name}_sync_anchors.json"
    if output_path.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing output without --force: {output_path}")
    live_path = args.live or preset.audio.live
    ref_path = args.ref or preset.audio.ref
    for label, path in (("live", live_path), ("reference", ref_path)):
        if not path.is_file():
            raise SystemExit(f"{label} audio does not exist: {path}")

    started = time.perf_counter()
    print(f"preset: {preset.name}")
    print(f"live:   {live_path}")
    print(f"ref:    {ref_path}")
    print("loading audio...")
    live_audio, _ = load_mono(live_path, target_sr=FS)
    ref_audio, _ = load_mono(ref_path, target_sr=FS)
    live_audio, live_start, live_end = crop_span(
        live_audio, start=args.live_start, end=args.live_end, label="live"
    )
    ref_audio, ref_start, ref_end = crop_span(
        ref_audio, start=args.ref_start, end=args.ref_end, label="reference"
    )
    print(
        f"shared span: live={live_start:.1f}..{live_end:.1f}s "
        f"ref={ref_start:.1f}..{ref_end:.1f}s"
    )

    live_tuning = estimate_tuning(live_audio, FS)
    ref_tuning = estimate_tuning(ref_audio, FS)
    print(f"tuning: live={live_tuning} cents ref={ref_tuning} cents")
    live_chroma, live_onset = extract_features(live_audio, live_tuning)
    ref_chroma, ref_onset = extract_features(ref_audio, ref_tuning)

    live_cens = quantized_chroma_to_CENS(live_chroma, 201, 50, FEATURE_RATE)[0]
    ref_cens = quantized_chroma_to_CENS(ref_chroma, 201, 50, FEATURE_RATE)[0]
    chroma_shift = compute_optimal_chroma_shift(live_cens, ref_cens)
    ref_chroma = shift_chroma_vectors(ref_chroma, chroma_shift)
    ref_onset = shift_chroma_vectors(ref_onset, chroma_shift)
    print(f"optimal chroma shift: {chroma_shift}")
    print("running high-resolution MrMsDTW...")

    path = sync_via_mrmsdtw(
        f_chroma1=live_chroma,
        f_onset1=live_onset,
        f_chroma2=ref_chroma,
        f_onset2=ref_onset,
        input_feature_rate=FEATURE_RATE,
        step_weights=STEP_WEIGHTS,
        threshold_rec=THRESHOLD_REC,
        verbose=False,
    )
    path = make_path_strictly_monotonic(path)
    dense_live_t = path[0].astype(np.float64) / FEATURE_RATE + live_start
    dense_ref_t = path[1].astype(np.float64) / FEATURE_RATE + ref_start
    qc = structural_qc(
        dense_live_t,
        dense_ref_t,
        window_seconds=args.qc_window,
        min_slope=args.qc_min_slope,
        max_slope=args.qc_max_slope,
    )
    print(
        "structural QC: " + ("PASS" if qc["passed"] else "FAIL")
        + f"  slope={qc['min_observed_slope']}..{qc['max_observed_slope']}"
    )
    for flag in qc["flags"]:
        print(
            f"  {flag['kind']}: live {flag['live_start']:.1f}.."
            f"{flag['live_end']:.1f}s -> ref {flag['ref_start']:.1f}.."
            f"{flag['ref_end']:.1f}s (slope={flag['slope']:.2f})"
        )
    if not qc["passed"] and not args.allow_qc_fail:
        raise SystemExit(
            "refusing to write a structurally implausible alignment; "
            "trim to the shared musical span or pass --allow-qc-fail for diagnosis"
        )

    candidates = sample_candidates(dense_live_t, dense_ref_t, args.n)
    dense_output = args.dense_output
    if dense_output is None:
        dense_output = output_path.with_name(output_path.stem + "_dense.npz")
    validation = None
    if args.validate:
        validation = validate(dense_live_t, dense_ref_t, args.validate)
    payload = {
        "live_file": live_path.name,
        "ref_file": ref_path.name,
        "sr": FS,
        "source": "sync_toolbox:mrmsdtw_chroma_dlnco",
        "synctoolbox_version": importlib.metadata.version("synctoolbox"),
        "feature_rate_hz": FEATURE_RATE,
        "shared_span": {
            "live_start": live_start,
            "live_end": live_end,
            "ref_start": ref_start,
            "ref_end": ref_end,
        },
        "sampling": {
            "anchor_count": args.n,
            "max_live_gap_s": float(np.max(np.diff([
                anchor["live_t"] for anchor in candidates
            ]))),
        },
        "structural_qc": qc,
        "validation": validation,
        "dense_path": None if args.no_dense else dense_output.name,
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "anchors": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)
    if not args.no_dense:
        write_dense_path(
            dense_output,
            live_t=dense_live_t,
            ref_t=dense_ref_t,
            feature_rate_hz=FEATURE_RATE,
        )
        print(f"wrote dense path -> {dense_output}")
    print(f"wrote {len(candidates)} anchors -> {output_path}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
