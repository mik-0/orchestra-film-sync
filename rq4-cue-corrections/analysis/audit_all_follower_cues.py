"""Independently check event timing, assignments and cut summaries in saved runs."""

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
import json
import hashlib
import numpy as np

HERE = _AUDIO_ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args(); checks=[]
    for method in ('dixon','arzt','concertcue'):
        path=args.directory/f'{method}.json';d=json.loads(path.read_text())
        with np.load(args.directory/f'{method}_paths.npz') as z:
            times=z['online_live_t'];paths={k[:-6]:z[k] for k in z.files if k.endswith('_ref_t') and k!='ground_truth_ref_t'}
        cuts=0;events=0
        for name,run in d['runs'].items():
            consumed=-1; previous_frame=-1; previous_estimate=0.
            assert len(run['events']) == (0 if name=='audio_only' else 22 if name=='perfect_all_cue_reset' else 13)
            for event in run['events']:
                i=int(np.searchsorted(times,event['event_time_s']-1e-12))
                assert times[i]==event['applied_at_audio_time_s']
                if i!=previous_frame:previous_estimate=float(paths[name][i-1]) if i else 0.
                assert abs(event['tracker_reference_before_s']-previous_estimate)<1e-10
                previous_frame=i
                choice=event['selected_cue_index']
                if name.startswith('automatic'):
                    width=float(name.split('width_')[1][:-1].replace('_','.'))
                    eligible=[j for j,m in enumerate(d['landmarks']) if j>consumed and abs(m['reference_s']-event['tracker_reference_before_s'])<=width+1e-12]
                    expected=min(eligible,key=lambda j:(abs(d['landmarks'][j]['reference_s']-event['tracker_reference_before_s']),j)) if eligible else None
                    assert choice==expected and eligible==event['eligible_cue_indices']
                if choice is not None:
                    assert choice>consumed;consumed=choice
                    m=d['landmarks'][choice]
                    previous_estimate=m['quantized_reference_s']
                    step=4096/44100 if method=='concertcue' else 1/30
                    assert -1e-10<=m['reference_s']-m['quantized_reference_s']<step+1e-10
                events+=1
        for hold,view in d['transition_sensitivity'].items():
            for row in view['rows']:
                for name,result in row['conditions'].items():
                    target=row['reference_target_s']; start=None; expected=None
                    for t,position in zip(times,paths[name]):
                        if position<target:start=None;continue
                        if start is None:start=t
                        if t-start>=float(hold)-1e-10:expected=float(t);break
                    # The stored cut error uniquely determines the cut time.
                    actual=result['signed_error_s']
                    if expected is None: assert actual is None
                    else:
                        assert abs(actual-(expected-row['marked_live_target_s']))<1e-10
                        assert result['acceptable']==(-.60<=actual<=.33)
                    cuts+=1
        primary=[r for r in d['transition_sensitivity']['0.0']['rows'] if r['included_primary']]
        for name in paths:
            raw=[r['conditions'][name]['signed_error_s'] for r in primary]
            es=np.array([v for v in raw if v is not None])
            cost=[]
            for e in es:
                tolerance=.60 if e<0 else .33
                cost.append(0 if -.60<=e<=.33 else 2 if abs(e)>=1 else (abs(e)-tolerance)/(1-tolerance))
            if len(es)==len(raw):
                assert abs(np.mean(cost)-d['score_primary_14'][name]['mean_cost'])<1e-12
            else:
                assert d['score_primary_14'][name]['mean_cost'] is None
                assert d['score_primary_14'][name]['unreached_targets']==len(raw)-len(es)
            assert int((np.abs(es)>=1).sum())==d['score_primary_14'][name]['catastrophes']
        if method=='arzt':
            with np.load(HERE/'output/published_firebird_matchmaker_arzt_trace.npz') as original:
                for hold,view in d['transition_sensitivity'].items():
                    for row in view['rows']:
                        start=None;expected=None
                        for t,pos in zip(original['online_live_t'],original['online_ref_t']):
                            if pos<row['reference_target_s']:start=None;continue
                            if start is None:start=t
                            if t-start>=float(hold)-1e-10:expected=float(t);break
                        assert abs(expected-row['marked_live_target_s']-row['conditions']['audio_only']['signed_error_s'])<1e-10
        checks.append({'historical_arzt_cut_times_unchanged':True if method=='arzt' else None,'method':method,'checked_cut_times':cuts,'checked_cue_events':events,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    output={'status':'passed','checks':checks,'scope':'Event application times, nearest-unused assignment, backward reference quantisation, all target crossing/hold times, acceptable flags and primary transition costs.', 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.directory/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
