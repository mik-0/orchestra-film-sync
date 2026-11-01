"""Audit and reanalyse stored Dixon error/energy traces without rerunning audio."""

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
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

HERE = _AUDIO_ROOT
REPOSITORY = _REPO_ROOT
sys.path.insert(0, str(HERE))
from online_dtw.alignment_reference import failure_episodes  # noqa: E402
from online_dtw.rank_bootstrap import moving_block_rank_correlation  # noqa: E402

PAIRS = ('firebird', 'chopin', 'yuja_fedorova', 'daphnis')
BLOCK_SECONDS = (10, 20, 30, 60)
LOW_ENERGY_RATIOS = (0.025, 0.05, 0.10, 0.20)
ERROR_LIMITS = (0.5, 1.0, 2.0)
SEED = 20260817
REPETITIONS = 10000


def file_record(path: Path) -> dict:
    return {'path': str(path.relative_to(REPOSITORY)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_trace(trace: dict, result: dict) -> dict:
    """Reject missing observations or a changed time convention before analysis."""
    for key in ('sample_t', 'sample_error_s', 'sample_energy_ratio',
                'online_live_t', 'error_s', 'availability_live_t'):
        values = trace[key]
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f'{key}: expected finite one-dimensional observations')
    t = trace['sample_t']
    if len(t) != len(trace['sample_error_s']) or len(t) != len(trace['sample_energy_ratio']):
        raise ValueError('unpaired analysis samples')
    if len(t) != result['energy_association']['n_samples']:
        raise ValueError('sample count differs from stored result')
    np.testing.assert_allclose(np.diff(t), 1.0, rtol=0, atol=1e-9)
    live_t = trace['online_live_t']
    if len(live_t) != len(trace['error_s']) or np.any(np.diff(live_t) <= 0):
        raise ValueError('invalid frame-level time series')
    np.testing.assert_array_equal(live_t, trace['availability_live_t'])
    expected_t = np.arange(np.ceil(live_t[0]), live_t[-1] + 1e-9, 1.0)
    np.testing.assert_array_equal(t, expected_t)
    np.testing.assert_allclose(trace['sample_error_s'],
                               np.interp(t, live_t, trace['error_s']), rtol=0, atol=1e-10)
    span = result['reference']['shared_span']
    if live_t[0] < span['live_start'] or live_t[-1] > span['live_end'] + 1e-9:
        raise ValueError('frame timestamps outside the approved shared span')
    energy = trace['sample_energy_ratio']
    if np.any(energy < 0):
        raise ValueError('negative RMS ratio')
    np.testing.assert_allclose(np.median(energy[energy > 0]), 1.0, rtol=0, atol=1e-10)
    return {'n_samples': len(t), 'sample_step_s': 1.0,
            'first_sample_s': float(t[0]), 'last_sample_s': float(t[-1]),
            'finite_regular_paired_samples': True, 'availability_time_verified': True,
            'positive_energy_median_verified': True}


def low_energy_counts(episodes: list, sample_t: np.ndarray,
                      energy: np.ndarray, ratio: float) -> dict:
    before_values = []
    missing = 0
    for episode in episodes:
        start = episode['live_start']
        before = energy[(sample_t >= start - 1.0) & (sample_t < start)]
        if before.size:
            before_values.append(float(np.mean(before)))
        else:
            missing += 1
    return {'episode_count': len(episodes), 'episodes_without_preceding_sample': missing,
            'episodes_with_preceding_sample': len(before_values),
            'episodes_after_low_energy': int(np.sum(np.asarray(before_values) <= ratio))}


def analyse_pair(pair: str) -> dict:
    result_path = HERE / 'output' / f'published_{pair}_matchmaker_dixon.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'final':
        raise ValueError(f'{pair} is not a final stored result')
    trace_path = REPOSITORY / result['provenance']['trace_path']
    with np.load(trace_path, allow_pickle=False) as stored:
        trace = {key: stored[key] for key in stored.files}
    validation = validate_trace(trace, result)
    original = result['energy_association']
    if (original['sample_step_s'], original['block_seconds'], original['repetitions']) != (1, 30, 1000):
        raise ValueError('stored protocol differs from expected original settings')
    x, y = np.abs(trace['sample_error_s']), trace['sample_energy_ratio']
    legacy = moving_block_rank_correlation(
        x, y, block_size=30, repetitions=1000, seed=SEED, legacy_fixed_ranks=True)
    np.testing.assert_allclose(
        [legacy.rho, legacy.lower, legacy.upper],
        [original['spearman_rho'], *original['moving_block_bootstrap_95_interval']],
        rtol=0, atol=1e-12)
    corrected = moving_block_rank_correlation(x, y, block_size=30, repetitions=1000, seed=SEED)
    np.testing.assert_allclose(corrected.rho, legacy.rho, rtol=0, atol=1e-12)
    print(f'{pair}: original reproduced; corrected 1000-resample interval '
          f'[{corrected.lower:.6f}, {corrected.upper:.6f}]', flush=True)
    correlations = []
    for seconds in BLOCK_SECONDS:
        estimate = moving_block_rank_correlation(
            x, y, block_size=seconds, repetitions=REPETITIONS, seed=SEED)
        correlations.append({'block_seconds': seconds, 'repetitions': REPETITIONS,
                             **asdict(estimate)})
        print(f'{pair}: {seconds}s blocks [{estimate.lower:.6f}, {estimate.upper:.6f}], '
              f'{estimate.undefined_resamples} undefined', flush=True)
    failures = []
    for threshold in ERROR_LIMITS:
        episodes = failure_episodes(trace['online_live_t'], trace['error_s'],
                                    threshold_s=threshold, minimum_duration_s=1.0)
        for ratio in LOW_ENERGY_RATIOS:
            counts = low_energy_counts(episodes, trace['sample_t'], y, ratio)
            if threshold == 1.0 and ratio == 0.10:
                expected = result['failure_summary']
                assert counts['episode_count'] == expected['episode_count']
                assert counts['episodes_after_low_energy'] == expected['episodes_starting_after_low_energy']
            failures.append({'error_limit_s': threshold, 'low_energy_ratio': ratio, **counts})
    return {'pair': result['pair'], 'inputs': [file_record(result_path), file_record(trace_path)],
            'validation': validation, 'legacy_reproduced': asdict(legacy),
            'corrected_same_resamples': asdict(corrected),
            'block_length_sensitivity': correlations, 'low_energy_sensitivity': failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    recordings = [analyse_pair(pair) for pair in PAIRS]
    aggregate = []
    for threshold in ERROR_LIMITS:
        for ratio in LOW_ENERGY_RATIOS:
            rows = [next(row for row in rec['low_energy_sensitivity']
                         if row['error_limit_s'] == threshold and row['low_energy_ratio'] == ratio)
                    for rec in recordings]
            aggregate.append({'error_limit_s': threshold, 'low_energy_ratio': ratio,
                              **{key: sum(row[key] for row in rows) for key in (
                                  'episode_count', 'episodes_without_preceding_sample',
                                  'episodes_with_preceding_sample', 'episodes_after_low_energy')}})
    baseline = next(row for row in aggregate
                    if row['error_limit_s'] == 1 and row['low_energy_ratio'] == 0.1)
    assert (baseline['episodes_after_low_energy'], baseline['episode_count']) == (4, 239)
    payload = {
        'status': 'completed_reanalysis', 'method': 'paired moving-block percentile bootstrap of Spearman correlation',
        'protocol': {'seed': SEED, 'block_seconds': BLOCK_SECONDS, 'repetitions': REPETITIONS,
                     'legacy_reproduction_repetitions': 1000, 'low_energy_ratios': LOW_ENERGY_RATIOS,
                     'error_limits_s': ERROR_LIMITS, 'minimum_failure_duration_s': 1.0,
                     'preceding_energy_window_s': 1.0, 'primary_block_seconds': 30},
        'recordings': recordings, 'aggregate_low_energy_sensitivity': aggregate,
        'limitations': ['Recordings are separate reporting units; pooled episode counts are descriptive.',
                       'Bootstrap coverage relies on temporal assumptions not established by this sensitivity check.',
                       'Long blocks contain few independently selected chunks in the short recordings.',
                       'Quiet means low relative RMS, not listener-annotated silence.'],
        'provenance': {
            'repository_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip(),
            'working_tree_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPOSITORY, text=True).splitlines(),
            'command': [sys.executable, *sys.argv], 'python': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'sources': [file_record(Path(__file__).resolve()),
                        file_record(HERE / 'online_dtw' / 'rank_bootstrap.py'),
                        file_record(HERE / 'online_dtw' / 'alignment_reference.py')]}}
    # Refuse any nonstandard NaN/Infinity JSON before creating the output directory.
    encoded = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'results.json').write_text(encoded)
    for name, rows in (
        ('correlations.csv', [{'pair': rec['pair'], **row} for rec in recordings
                              for row in rec['block_length_sensitivity']]),
        ('low_energy_counts.csv', [{'pair': rec['pair'], **row} for rec in recordings
                                   for row in rec['low_energy_sensitivity']]),
        ('aggregate_low_energy_counts.csv', aggregate),
    ):
        with (args.output_dir / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Wrote audited reanalysis to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
