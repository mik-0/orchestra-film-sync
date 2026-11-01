"""Apply the frozen Firebird cue protocol to Dixon, Arzt and ConcertCue.

Retains original results. Uses validated Chapter 6 feature caches and published
frame availability times. All cue conditions use the same annotation/detection
inputs, reference-position gates, target exclusions and causal cut rules.
"""

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

from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import time
import numpy as np

HERE = _AUDIO_ROOT
ROOT = _REPO_ROOT
sys.path.insert(0, str(HERE))
import analyse_all_followers as study
import run_published_tracker as published
import visual_all_cue_resets as protocol
import summarize_visual_contribution as scoring

METHODS = {'dixon': 'external_matchmaker_dixon_v0_3_0',
           'arzt': 'external_matchmaker_arzt_v0_3_0',
           'concertcue': 'external_concertcue_otw_2876aaca'}


class Decoder:
    def __init__(self, method, ref, offset, remaining, *, full_cc=False):
        self.method, self.offset = method, offset
        self.last = 0.
        if method == 'concertcue':
            cc = study.concertcue()
            import libaudio_rt.otw as module
            if not full_cc: module.np = study.PrefixAllocation(remaining)
            try: self.follower = cc.OTW(ref[:, offset:], study.PARAMS)
            finally: module.np = np
            self.ref_step = 4096 / 44100
        else:
            self.follower, _, _ = published.make_matchmaker_follower(
                ref[offset:], method=METHODS[method], frame_rate=30)
            self.ref_step = 1 / 30

    def advance(self, observation, frame_index):
        if self.method == 'concertcue':
            self.last = float(self.follower.insert(observation)) * self.ref_step
        elif self.follower.is_still_following():
            self.last = float(self.follower(observation, frame_index / 30))
        return self.offset * self.ref_step + self.last

    def clear(self):
        protocol.clear_follower(self.follower)


def decode(method, ref, live, times, landmarks, emissions, mode, width=None, full_cc=False):
    n=len(times); decoder=Decoder(method,ref,0,n,full_cc=full_cc)
    positions=np.empty(n); events=[]; selected=[]; skipped=[]
    cursor=0; consumed=-1; previous=0.
    inputs=[] if mode=='audio_only' else landmarks if mode=='annotated' else emissions
    for i, available in enumerate(times):
        timekey='cue_time_s' if mode=='annotated' else 'time_s'
        while cursor<len(inputs) and inputs[cursor][timekey] <= available+1e-12:
            entry=inputs[cursor]
            if mode=='annotated': choice=cursor; eligible=[choice]
            else:
                choice,eligible=protocol.select_eligible_landmark(previous,landmarks,
                        last_consumed_index=consumed,width_s=width)
            event={'emission_index':int(entry.get('emission_index',cursor)),
                   'event_time_s':float(entry[timekey]),'applied_at_audio_time_s':float(available),
                   'audio_application_delay_s':float(available-entry[timekey]),
                   'tracker_reference_before_s':previous,'eligible_cue_indices':eligible,
                   'selected_cue_index':choice}
            if choice is not None:
                landmark=landmarks[choice]
                skipped.extend(range(consumed+1,choice))
                decoder.clear()
                decoder=Decoder(method,ref,landmark['reference_frame'],n-i,full_cc=full_cc)
                previous=landmark['quantized_reference_s']
                selected.append(choice);consumed=choice
                event['reset_reference_s']=previous
            events.append(event);cursor+=1
        obs=live[:,i] if method=='concertcue' else live[i]
        previous=decoder.advance(obs,i);positions[i]=previous
    decoder.clear()
    return positions, {'events':events,'n_resets':len(selected),'n_ignored_emissions':len(events)-len(selected),
                       'selected_cue_indices':selected,'skipped_before_selected_cue_indices':skipped}


from score_transitions import costs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-cache',type=Path,default=ROOT/'tmp/audio-failure-all-methods')
    p.add_argument('--output-dir',type=Path,default=HERE/'output/cue_all_followers_20261026')
    p.add_argument('--resume',action='store_true',help='Retain completed follower outputs and run missing followers')
    args=p.parse_args()
    if args.output_dir.exists() and not args.resume:raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    study.retain_arzt_array()
    olddir=HERE/'output/visual_all_cue_resets_v1'
    corrected=HERE/'output/visual_all_cue_resets_v2_causal_holds/results.json'
    old=json.loads(corrected.read_text())
    old_arrays=np.load(olddir/'causal_paths.npz')
    detector=json.loads((olddir/'held_out_detector.json').read_text())
    saved_landmarks=json.loads((olddir/'cue_landmarks.json').read_text())['landmarks']
    base_rows=old['transition_sensitivity']['0.0']['rows']
    transitions=[protocol.Transition(r['segment'],r['marked_live_target_s'],r['reference_target_s']) for r in base_rows]
    paired={r['segment'] for r in base_rows if r['has_approved_pair']}
    index=json.loads(study.INDEX.read_text())['rows']
    summary={'scope':'All three followers on the same Firebird cue experiment', 'methods':{},
        'protocol':{'assignment_ranges_s':protocol.ELIGIBILITY_WIDTHS_S,'holds_s':protocol.STABILITY_HOLDS_S,
          'cue_count':len(saved_landmarks),'detection_count':len(detector['emissions']),
          'excluded_targets':sorted(protocol.PRIMARY_EXCLUSIONS),
          'reset':'At the first available frame at or after the event; fresh decoder, retained precomputed feature history',
          'reference_quantisation':'latest reference frame not after the annotated reference position',
          'automatic_assignment':'nearest unused reference cue within range, using the last available estimate; skip earlier cues',
          'timing':'each follower retains its own published live availability and reference frame spacing',
          'detector_retrained':False,'parameter_tuning':False},
        'provenance':study.provenance(),
        'inputs':[study.record(x) for x in (corrected,olddir/'causal_paths.npz',olddir/'cue_landmarks.json',olddir/'held_out_detector.json',Path(__file__).resolve(),Path(protocol.__file__),Path(scoring.__file__))]}
    if args.resume:
        previous=json.loads((args.output_dir/'summary.json').read_text())
        assert previous['protocol']==json.loads(json.dumps(summary['protocol']))
        for inp in previous['inputs']:
            if inp['path'].endswith('run_all_follower_cue_corrections.py'):
                assert study.record(args.output_dir/'runner_initial.py')['sha256']==inp['sha256']
            else: assert published.sha256(ROOT/inp['path'])==inp['sha256']
        summary['methods']=previous['methods']
        summary['resumed_from']=previous['provenance']
        summary['inputs'].append(study.record(args.output_dir/'runner_initial.py'))
    for method in METHODS:
        if method in summary['methods']:
            print('KEEP COMPLETED',method,flush=True)
            continue
        start=time.perf_counter();print('START',method,flush=True)
        family='cc' if method=='concertcue' else 'matchmaker'
        feature_path=args.feature_cache/f'firebird_{family}.npz'
        metadata=json.loads(feature_path.with_suffix('.json').read_text())
        with np.load(feature_path) as z:ref=z['ref'];live=z['live']
        source=next(r for r in index if r['pair']=='firebird' and r['method']==METHODS[method])
        result,trace,inputs=study.load_case(source)
        assert metadata['span']==result['reference']['shared_span']
        assert metadata['feature_source_sha256']==published.sha256(Path(published.__file__))
        times=trace['online_live_t']
        if method=='dixon':
            np.testing.assert_allclose(times,old_arrays['online_live_t'],atol=1e-10,rtol=0)
            times=old_arrays['online_live_t']
        assert len(times)==(live.shape[1] if method=='concertcue' else len(live))
        step=4096/44100 if method=='concertcue' else 1/30
        nref=ref.shape[1] if method=='concertcue' else len(ref)
        landmarks=[]
        for oldmark in saved_landmarks:
            mark=dict(oldmark);frame=int(np.clip(np.floor(mark['reference_s']/step+1e-12),0,nref-1))
            mark.update(reference_frame=frame,quantized_reference_s=frame*step);landmarks.append(mark)
        paths={};runs={}
        conditions=[('audio_only','audio_only',None),('perfect_all_cue_reset','annotated',None)]
        conditions += [(protocol.condition_for_width(w),'automatic',w) for w in protocol.ELIGIBILITY_WIDTHS_S]
        for name,mode,width in conditions:
            paths[name],runs[name]=decode(method,ref,live,times,landmarks,detector['emissions'],mode,width)
            assert np.isfinite(paths[name]).all()
            print(method,name,'resets',runs[name]['n_resets'],flush=True)
        diff=np.abs(paths['audio_only']-trace['online_ref_t'])
        validation={'historical_baseline_max_difference_s':float(diff.max()),
                    'historical_baseline_different_frames':int((diff>1e-5).sum())}
        if method=='arzt':
            assert (diff>1e-5).sum()<=2 and diff.max()<=1/30+1e-5
        else:np.testing.assert_allclose(paths['audio_only'],trace['online_ref_t'],atol=1e-5,rtol=0)
        if method=='dixon':
            np.testing.assert_array_equal(times,old_arrays['online_live_t'])
            for name in paths:np.testing.assert_allclose(paths[name],old_arrays[name+'_ref_t'],atol=1e-5,rtol=0)
            validation['all_six_historical_dixon_paths_reproduced']=True
        if method=='concertcue':
            # Full original allocation independently checks the storage-only adaptation.
            for name,mode,width in conditions:
                alternative,_=decode(method,ref,live,times,landmarks,detector['emissions'],mode,width,full_cc=True)
                np.testing.assert_array_equal(alternative,paths[name])
            validation['all_six_paths_match_original_full_allocation']=True
        views={}
        for hold in protocol.STABILITY_HOLDS_S:
            rows=protocol.transition_rows(transitions,times,paths,paired_segments=paired,hold_s=hold)
            views[str(hold)]={'rows':rows,'summary':protocol.summarize_rows(rows,list(paths))}
        if method=='dixon':
            for hold,view in views.items():assert view['rows']==old['transition_sensitivity'][hold]['rows']
            validation['all_dixon_transition_rows_and_holds_reproduced']=True
        main=[r for r in views['0.0']['rows'] if r['included_primary']]
        # Existing score sensitivity: the exact grid is recorded in the output.
        cost_sensitivity=[]
        for threshold in (.75,1.,1.5,2.):
            for penalty in (0.,.5,1.,2.):
                cost_sensitivity.append({'threshold_s':threshold,'penalty':penalty,
                  'primary_14':costs(main,paths,threshold,penalty),
                  'all_16':costs(views['0.0']['rows'],paths,threshold,penalty)})
        payload={'method':method,'validation':validation,'landmarks':landmarks,
          'runs':runs,'assignment_audit':protocol.assignment_audit({k:v for k,v in runs.items() if k.startswith('automatic')},detector),
          'transition_sensitivity':views,'score_primary_14':costs(main,paths),
          'score_all_16':costs(views['0.0']['rows'],paths),'score_sensitivity':cost_sensitivity,
          'dense_metrics':{k:protocol.dense_metrics(v-trace['ground_truth_ref_t']) for k,v in paths.items()},
          'inputs':inputs+[study.record(feature_path),study.record(feature_path.with_suffix('.json'))],
          'feature_metadata':metadata,'seconds':time.perf_counter()-start}
        study.write(args.output_dir/f'{method}.json',payload)
        np.savez_compressed(args.output_dir/f'{method}_paths.npz',online_live_t=times,
                            ground_truth_ref_t=trace['ground_truth_ref_t'],**{k+'_ref_t':v for k,v in paths.items()})
        summary['methods'][method]={'validation':validation,
          'primary':views['0.0']['summary']['primary_14'], 'scores':payload['score_primary_14'],
          'holds':{k:v['summary']['primary_14']['conditions'] for k,v in views.items()}}
        study.write(args.output_dir/'summary.json',summary)
        print('DONE',method,json.dumps(summary['methods'][method]['primary']['conditions']),flush=True)


if __name__=='__main__':main()
