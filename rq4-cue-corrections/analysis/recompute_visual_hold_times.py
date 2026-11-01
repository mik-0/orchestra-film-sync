#!/usr/bin/env python3
"""Correct hold-trigger timing on frozen all-cue paths without rerunning tracking."""
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
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from summarize_visual_contribution import repository_path, sha256
from visual_all_cue_resets import Transition, summarize_rows, transition_rows

PROJECT = _AUDIO_ROOT
ROOT = _REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=PROJECT / 'output/visual_all_cue_resets_v1')
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT / 'output/visual_all_cue_resets_v2_causal_holds')
    args = parser.parse_args()
    output = args.output_dir / 'results.json'
    if output.exists():
        raise SystemExit(f'refusing to overwrite {output}')
    result_path = args.source_dir / 'results.json'
    trace_path = args.source_dir / 'causal_paths.npz'
    source_hashes = {path: sha256(path) for path in args.source_dir.iterdir() if path.is_file()}
    original = json.loads(result_path.read_text())
    original_rows = original['transition_sensitivity']['0.0']['rows']
    names = list(original_rows[0]['conditions'])
    transitions = [Transition(row['segment'], row['marked_live_target_s'],
                              row['reference_target_s']) for row in original_rows]
    paired = {row['segment'] for row in original_rows if row['has_approved_pair']}
    with np.load(trace_path) as arrays:
        live_t = arrays['online_live_t']
        paths = {name: arrays[name + '_ref_t'] for name in names}
    if not np.all(np.isfinite(live_t)) or not np.all(np.diff(live_t) > 0):
        raise ValueError('saved observation times must be finite and increasing')
    if any(path.shape != live_t.shape or not np.all(np.isfinite(path)) for path in paths.values()):
        raise ValueError('saved paths must be finite and match the time grid')
    corrected = {}
    for key, view in original['transition_sensitivity'].items():
        rows = transition_rows(transitions, live_t, paths, paired_segments=paired,
                               hold_s=float(view['hold_s']))
        corrected[key] = {'hold_s': view['hold_s'], 'rows': rows,
                          'summary': summarize_rows(rows, names)}
    if corrected['0.0'] != original['transition_sensitivity']['0.0']:
        raise RuntimeError('the primary first-crossing results changed')
    payload = copy.deepcopy(original)
    payload['schema_version'] = 2
    payload['transition_sensitivity'] = corrected
    payload['transition_summary'] = corrected['0.0']['summary']
    payload['method']['transition_fire_rule'] = (
        'First output at or beyond the target for zero hold. For positive holds, '
        'the first output after the full elapsed duration at or beyond the target; '
        'any output below the target cancels the wait. No backdating.'
    )
    payload['method']['hold_duration_rounding_tolerance_s'] = 1e-10
    payload['trace_generation_provenance'] = original['provenance']
    inputs = [result_path, trace_path, Path(__file__),
              _script_path('visual_all_cue_resets.py'),
              _script_path('visual_paired_cue_resets.py')]
    payload['provenance'] = {
        'operation': 'recompute transition triggers on frozen estimate paths',
        'command': [sys.executable, repository_path(Path(__file__)), *sys.argv[1:]],
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_has_changes': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        'inputs': [{'path': repository_path(path), 'sha256': sha256(path)} for path in inputs],
        'python_version': sys.version.split()[0], 'numpy_version': np.__version__,
        'tracker_rerun': False,
    }
    payload['hold_recomputation'] = {
        'supersedes': 'positive-hold trigger times and their derived summaries only',
        'primary_first_crossing_view_unchanged': True,
        'original_result_files_unchanged': True,
        'limits': ('Targets are evaluated independently as in the original analysis. '
                   'No compute or display latency is added. This does not re-audit '
                   'the tracker, detector, or reference construction.'),
    }
    if source_hashes != {path: sha256(path) for path in source_hashes}:
        raise RuntimeError('a frozen source artifact changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'wrote {output}')
    for hold, view in corrected.items():
        values = view['summary']['primary_14']['conditions']
        print(hold, {name: values[name]['acceptable_count'] for name in names})


if __name__ == '__main__':
    main()
