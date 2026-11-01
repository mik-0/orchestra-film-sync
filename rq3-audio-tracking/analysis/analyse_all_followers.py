"""Extend the frozen audio-failure study to all three published followers.

Trace analyses reuse saved estimates. Restarts replay each follower at its own
merged failure peaks, with the same quantised oracle position for the follower
and clock. Full remaining reference features are available. Feature histories
are retained; decoder alignment state is fresh. Original artifacts are untouched.
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
import os
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

HERE = _AUDIO_ROOT
ROOT = _REPO_ROOT
sys.path.insert(0, str(HERE))
import run_published_tracker as published
from run_completion_probe import retain_arzt_array
from audio_failure_analysis import merge_failure_bouts, episode_duration_summary
from reanalyse_audio_energy import validate_trace, low_energy_counts
from online_dtw.alignment_reference import failure_episodes
from online_dtw.rank_bootstrap import moving_block_rank_correlation
from online_dtw.presets import Preset

OUT = HERE / 'output/audio_failure_all_methods_v1'
CACHE = ROOT / 'tmp/audio-failure-all-methods'
INDEX = HERE / 'output/benchmark-completion-20261025/completed_comparison.json'
LEADS = (2., 5., 10., 20., 40.)
LIMITS = (.5, 1., 2.)
BLOCKS = (10, 20, 30, 60)
QUIET = (.025, .05, .1, .2)
SEED = 20260817
PARAMS = {'c': 300, 'max_run_count': 3, 'diag_weight': .4}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def record(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': published.sha256(path)}


def provenance():
    return {'command': sys.argv, 'revision': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script': record(Path(__file__).resolve()),
        'dependencies': [record(Path(m.__file__).resolve()) for m in
                         (published, sys.modules['audio_failure_analysis'],
                          sys.modules['online_dtw.rank_bootstrap'])],
        'python': sys.version, 'numpy': np.__version__,
        'pymatchmaker': importlib.metadata.version('pymatchmaker')}


def load_case(row):
    path = ROOT / row['source']
    result = json.loads(path.read_text())
    trace_path = path.with_name(path.stem+'_trace.npz')
    with np.load(trace_path) as z:
        trace = {k: z[k] for k in z.files}
    validate_trace(trace, result)
    for field, hashfield in [('path', 'sha256'), ('dense_path', 'dense_sha256')]:
        assert published.sha256(ROOT/result['reference'][field]) == result['reference'][hashfield]
    assert result['metrics']['median_abs_error_s'] == row['median_abs_error_s']
    return result, trace, [record(path), record(trace_path)]


def trace_analysis(row, result, trace, inputs):
    timepoints, errors = trace['online_live_t'], trace['error_s']
    sensitivities = []
    for limit in LIMITS:
        episodes = failure_episodes(timepoints, errors, threshold_s=limit, minimum_duration_s=1.)
        duration = sum(e['duration_s'] for e in episodes)
        span = result['reference']['shared_span']
        quiet = [{'ratio': q, **low_energy_counts(episodes, trace['sample_t'],
                    trace['sample_energy_ratio'], q)} for q in QUIET]
        sensitivities.append({'error_limit_s': limit, 'episodes': episodes,
            'count': len(episodes), 'peak_behind_count': sum(e['peak_error_s'] < 0 for e in episodes),
            'duration_s': duration, 'time_fraction': duration/float(timepoints[-1]-timepoints[0]),
            'duration_summary': episode_duration_summary(episodes), 'quiet_counts': quiet})
    primary = sensitivities[1]
    assert primary['count'] == result['failure_summary']['episode_count']
    np.testing.assert_allclose(primary['time_fraction'], row['failure_time_fraction'], atol=1e-12)
    correlations = []
    for block in BLOCKS:
        b = moving_block_rank_correlation(np.abs(trace['sample_error_s']),
                trace['sample_energy_ratio'], block_size=block, repetitions=10000, seed=SEED)
        correlations.append({'block_seconds': block, **asdict(b)})
        print(row['pair'], row['method'], 'bootstrap', block, b.lower, b.upper, flush=True)
    bouts = merge_failure_bouts(timepoints, errors, primary['episodes'])
    return {'pair': row['pair'], 'method': row['method'], 'inputs': inputs,
            'metrics': result['metrics'], 'failure_sensitivity': sensitivities,
            'correlations': correlations, 'bouts': bouts, 'provenance': provenance(),
            'protocol': {'error_limits_s': LIMITS, 'minimum_duration_s': 1.,
                'merge_gap_s': 5., 'quiet_ratios': QUIET, 'preceding_window_s': 1.,
                'bootstrap_repetitions': 10000, 'seed': SEED, 'blocks_s': BLOCKS}}


def concertcue():
    archive = published.CACHE/'web-score-following/web/public/python/libaudio_rt.zip'
    assert published.sha256(archive) == published.CONCERTCUE_ARCHIVE_SHA256
    js = types.ModuleType('js'); js.Blob = js.document = js.window = None
    sys.modules.setdefault('js', js)
    if str(archive) not in sys.path:
        sys.path.insert(0, str(archive))
    import libaudio_rt
    return libaudio_rt


class PrefixAllocation:
    """Allocate only columns that this finite replay can visit; all rows remain."""
    def __init__(self, frames):
        self.frames = frames

    def __getattr__(self, key):
        return getattr(np, key)

    def full(self, shape, fill_value, dtype):
        return np.full((shape[0], self.frames), fill_value, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros((shape[0], self.frames), dtype=dtype)


def decode_cc(ref, live, prefix=True):
    cc = concertcue()
    import libaudio_rt.otw as module
    # The original code reads D only through the current live frame t.
    # Restricting preallocated future columns does not truncate its search.
    if prefix:
        module.np = PrefixAllocation(live.shape[1])
    try:
        follower = cc.OTW(ref, PARAMS)
    finally:
        module.np = np
    positions = np.array([follower.insert(live[:, i]) for i in range(live.shape[1])])
    return positions*4096/44100


def features(pair, result, args):
    family = 'cc' if 'concertcue' in result['method'] else 'matchmaker'
    cache = CACHE/f'{pair}_{family}.npz'
    preset = Preset.load(pair)
    if pair == 'firebird':
        preset.audio.ref = args.firebird_reference
    span = result['reference']['shared_span']
    media = {side: {'file': path.name, 'sha256': published.sha256(path)}
             for side, path in [('ref', preset.audio.ref), ('live', preset.audio.live)]}
    metadata = {'media': media, 'span': span, 'family': family,
                'feature_source_sha256': published.sha256(Path(published.__file__))}
    if cache.exists():
        assert json.loads(cache.with_suffix('.json').read_text()) == metadata
        with np.load(cache) as z:
            return z['ref'], z['live'], metadata
    extracted = {}
    for side, path in [('ref', preset.audio.ref), ('live', preset.audio.live)]:
        audio = published.load_span_direct(path, start=span[side+'_start'], end=span[side+'_end'], sr=44100)
        if family == 'matchmaker':
            from matchmaker.features.audio import LogSpectralEnergyProcessor
            feat = published.compute_matchmaker_features_chunked(audio,
                processor=LogSpectralEnergyProcessor(sample_rate=44100, hop_length=1470),hop=1470)
        else:
            cc = concertcue()
            if side == 'ref':
                feat = cc.audio_to_np_cens(audio,44100,8192,4096)
            else:
                maker = cc.ChromaMaker(44100,8192)
                feat = np.empty((12,(len(audio)-8192)//3686+1))
                for i in range(feat.shape[1]):
                    feat[:,i] = maker.insert(audio[i*3686:i*3686+8192])
        extracted[side] = feat
        del audio
        print(pair, family, side, 'features',feat.shape,flush=True)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez(cache,**extracted);write(cache.with_suffix('.json'),metadata)
    return extracted['ref'],extracted['live'],metadata


def restart_one(result, trace, ref, live, bout, lead):
    t = trace['online_live_t']; span = result['reference']['shared_span']
    peak = bout['peak_live_t']; desired = peak-lead
    row = {'lead_s':lead,'target_s':peak,'requested_start_s':desired,'available':False}
    if desired < t[0]-1e-8:
        return {**row,'reason':'insufficient preceding recording'}
    start = int(np.searchsorted(t,desired-1e-8));end = int(np.argmin(np.abs(t-peak)))
    assert abs(t[end]-peak)<1e-8 and start<=end
    actual_start = float(t[start]);actual_end = float(t[end])
    truth = float(np.interp(actual_start,t,trace['ground_truth_ref_t']))
    target_truth = float(trace['ground_truth_ref_t'][end])
    is_cc = 'concertcue' in result['method']
    ref_step = 4096/44100 if is_cc else 1/30
    ref_len = ref.shape[1] if is_cc else ref.shape[0]
    ref_index = int(np.clip(round((truth-span['ref_start'])/ref_step),0,ref_len-1))
    supplied_position = span['ref_start']+ref_index*ref_step
    if is_cc:
        positions = decode_cc(ref[:,ref_index:],live[:,start:end+1])
    else:
        positions,_,_,_ = published.decode_matchmaker(ref[ref_index:],live[start:end+1],
                    method=result['method'],frame_rate=30,progress=False)
    estimated = float(positions[-1]+supplied_position)
    clock = supplied_position+actual_end-actual_start
    assert np.isfinite(positions).all()
    return {**row,'available':True,'actual_start_s':actual_start,'actual_lead_s':actual_end-actual_start,
        'live_start_frame':start,'live_end_frame':end,'reference_start_frame':ref_index,
        'supplied_reference_s':supplied_position,'oracle_quantisation_error_s':supplied_position-truth,
        'target_reference_s':target_truth,'estimate_reference_s':estimated,'clock_reference_s':clock,
        'restart_signed_error_s':estimated-target_truth,'restart_abs_error_s':abs(estimated-target_truth),
        'clock_abs_error_s':abs(clock-target_truth),'processed_frames':end-start+1}


def summarize(rows):
    output=[]
    for lead in LEADS:
        kept=[r for r in rows if r['available'] and r['lead_s']==lead]
        n=len(kept)
        output.append({'lead_s':lead,'n':n,
            'success_count':sum(r['restart_abs_error_s']<1 for r in kept),
            'clock_success_count':sum(r['clock_abs_error_s']<1 for r in kept),
            'better_than_clock_count':sum(r['restart_abs_error_s']<r['clock_abs_error_s'] for r in kept),
            'median_abs_error_s':float(np.median([r['restart_abs_error_s'] for r in kept])) if n else None})
    return output


def worker_init(row, args):
    global WORKER
    retain_arzt_array()
    result,trace,_=load_case(row)
    ref,live,_=features(row['pair'],result,args)
    WORKER=(result,trace,ref,live)


def worker_bout(item):
    i,bout=item
    result,trace,ref,live=WORKER
    return [{'bout':i,**restart_one(result,trace,ref,live,bout,lead)} for lead in LEADS]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['traces','restarts'],required=True)
    parser.add_argument('--pair')
    parser.add_argument('--method',choices=['arzt','dixon','concertcue'])
    parser.add_argument('--firebird-reference',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    assert importlib.metadata.version('pymatchmaker')=='0.3.0'
    retain_arzt_array()
    for row in json.loads(INDEX.read_text())['rows']:
        method=next(m for m in ['arzt','dixon','concertcue'] if m in row['method'])
        pair=row['pair'];key=f'{pair}_{method}'
        if args.pair and args.pair!=pair or args.method and args.method!=method:continue
        target=OUT/f'{key}_{args.phase}.json'
        if target.exists():
            print('preserving completed',target.name,flush=True);continue
        result,trace,inputs=load_case(row)
        if args.phase=='traces':
            payload=trace_analysis(row,result,trace,inputs)
        else:
            analysis=json.loads((OUT/f'{key}_traces.json').read_text())
            ref,live,meta=features(pair,result,args)
            assert (live.shape[1] if method=='concertcue' else len(live))==len(trace['online_live_t'])
            checkpoint=OUT/f'{key}_restart_checkpoint.json'
            previous=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'completed_bouts':0,'rows':[]}
            rows=previous['rows'];completed=previous['completed_bouts']
            assert len(rows)==completed*len(LEADS)
            jobs=list(enumerate(analysis['bouts']))[completed:]
            resumed=record(checkpoint) if checkpoint.exists() else None
            def save_bout(new_rows):
                rows.extend(new_rows)
                count=len(rows)//len(LEADS)
                write(checkpoint,{'completed_bouts':count,'rows':rows})
                print(key,'restart',count,'/',len(analysis['bouts']),flush=True)
            if args.workers>1 and jobs:
                with ProcessPoolExecutor(max_workers=args.workers,
                        mp_context=multiprocessing.get_context('spawn'),
                        initializer=worker_init,initargs=(row,args)) as executor:
                    for batch in executor.map(worker_bout,jobs):save_bout(batch)
            else:
                for i,bout in jobs:
                    save_bout([{'bout':i,**restart_one(result,trace,ref,live,bout,lead)} for lead in LEADS])
            payload={'pair':pair,'method':method,'inputs':inputs,'features':meta,
                'protocol':{'leads_s':LEADS,'failure_selection':'each method own merged episodes',
                    'start':'first available frame at or after requested restart',
                    'reference_start':'nearest reference frame to offline truth at actual restart',
                    'clock_start':'same quantised reference position and actual live start',
                    'feature_history':'preserved, decoder alignment state reset',
                    'reference_extent':'all remaining reference frames',
                    'concertcue_storage':'only live columns needed by replay allocated; all reference rows retained'},
                'rows':rows,'summary':summarize(rows),'provenance':provenance(),
                'resumed_checkpoint':resumed,'workers':args.workers}
            del ref,live
        write(target,payload)
        print('saved',target.name,flush=True)


if __name__=='__main__':main()
