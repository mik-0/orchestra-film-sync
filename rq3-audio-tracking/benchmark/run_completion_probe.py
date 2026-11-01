"""Investigate missing benchmark cases without replacing published artifacts.

Uses the pinned runner and its scoring. Arzt retains the original NumPy cost
array between calls to avoid destruction of a long-lived Cython view. ConcertCue
keeps its original recurrence and dense float32 matrix, backed by a temporary
file. Tracking and calibration matrices are released sequentially.

Pass the same arguments as run_published_tracker.py. The output must be under
output/benchmark-completion-20261025/. Results remain provisional.
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


import gc
import json
import mmap
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

import numpy as np

import run_published_tracker as baseline

INVESTIGATION = baseline.PROJECT_ROOT / 'output/benchmark-completion-20261025'
TEMP_ROOT = baseline.REPOSITORY_ROOT / 'tmp/benchmark-completion'


def retain_arzt_array() -> None:
    original_factory = baseline.make_matchmaker_follower

    def factory(*args, **kwargs):
        follower, name, config = original_factory(*args, **kwargs)
        if name == 'OnlineTimeWarpingArztFrame':
            owner = follower.global_cost_matrix
            step = follower.step

            def owned_step(observation):
                step(observation)
                if not np.shares_memory(np.asarray(follower.global_cost_matrix), owner):
                    raise RuntimeError('Arzt unexpectedly replaced its cost buffer')
                follower.global_cost_matrix = owner

            follower.step = owned_step
        return follower, name, config

    baseline.make_matchmaker_follower = factory


class DiskAllocation:
    def __init__(self, path: Path):
        self.path = path
        self.mapping = None

    def __getattr__(self, name):
        return getattr(np, name)

    def full(self, shape, fill_value, dtype):
        required = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if shutil.disk_usage(self.path.parent).free < required + 5 * 1024**3:
            raise RuntimeError('Insufficient disk space for matrix plus 5 GiB reserve')
        result = np.memmap(self.path, mode='w+', shape=shape, dtype=dtype)
        rows = max(1, (128 * 1024**2) // (shape[1] * result.dtype.itemsize))
        for start in range(0, shape[0], rows):
            result[start:start + rows] = fill_value
            result.flush()
            if hasattr(result._mmap, 'madvise'):
                result._mmap.madvise(mmap.MADV_DONTNEED)
        print(f'Disk-backed matrix: {required / 1024**3:.2f} GiB', flush=True)
        self.mapping = result
        return np.asarray(result)


def install_concertcue_storage(directory: Path) -> None:
    archive = baseline.CACHE / 'web-score-following/web/public/python/libaudio_rt.zip'
    if baseline.sha256(archive) != baseline.CONCERTCUE_ARCHIVE_SHA256:
        raise RuntimeError('ConcertCue archive does not match pinned hash')
    js = types.ModuleType('js')
    js.Blob = js.document = js.window = None
    sys.modules.setdefault('js', js)
    sys.path.insert(0, str(archive))
    import libaudio_rt
    import libaudio_rt.otw as module

    original = libaudio_rt.OTW
    previous = None
    count = 0

    def factory(ref, params):
        nonlocal previous, count
        # The baseline creates calibration only after it saves all tracking
        # positions into a separate ndarray. It never reads the first D again.
        if previous is not None:
            previous.flush()
            filename = Path(previous.filename)
            previous._mmap.close()
            filename.unlink()
            previous = None
            gc.collect()
        count += 1
        allocator = DiskAllocation(directory / f'cost-{count}.dat')
        module.np = allocator
        try:
            follower = original(ref, params)
        finally:
            module.np = np
        previous = allocator.mapping
        original_insert = follower.insert
        frames = 0
        started = time.monotonic()
        stage = count

        def insert_with_progress(observation):
            nonlocal frames
            position = original_insert(observation)
            frames += 1
            if frames % 5000 == 0:
                print(f'Stage {stage}: {frames} frames, {time.monotonic() - started:.1f}s', flush=True)
            return position

        follower.insert = insert_with_progress
        return follower

    libaudio_rt.OTW = factory
    # Override the old resource gate for this separate, disk-backed probe only.
    baseline.CONCERTCUE_MAX_MATRIX_BYTES = float('inf')


def main() -> None:
    probe_hash = baseline.sha256(Path(__file__))
    args = baseline.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(INVESTIGATION.resolve()):
        raise SystemExit(f'Probe outputs must be under {INVESTIGATION}')
    if args.trace_output and not args.trace_output.resolve().is_relative_to(INVESTIGATION.resolve()):
        raise SystemExit('Probe trace must also stay in the investigation directory')
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='concertcue-', dir=TEMP_ROOT) as scratch:
        if args.method == 'external_matchmaker_arzt_v0_3_0':
            retain_arzt_array()
            adjustment = 'Retain original NumPy cost-array owner between Arzt steps.'
        elif args.method == 'external_concertcue_otw_2876aaca':
            install_concertcue_storage(Path(scratch))
            adjustment = 'Dense float32 cost matrix on disk; release tracking matrix before calibration.'
        else:
            raise SystemExit('This probe supports only Arzt and ConcertCue')
        baseline.main()
    data = json.loads(output.read_text())
    data['completion_investigation'] = {
        'status': 'provisional_not_integrated_into_thesis',
        'date': '2026-10-25',
        'resource_adjustment': adjustment,
        'published_artifacts_replaced': False,
        'original_runner_sha256': baseline.sha256(Path(baseline.__file__)),
        'probe_sha256': probe_hash,
        'arguments': sys.argv[1:],
    }
    output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
