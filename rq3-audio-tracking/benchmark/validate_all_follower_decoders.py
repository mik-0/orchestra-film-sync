"""Check replay adapters against the original decoders and frozen outputs."""

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
import json
from pathlib import Path

import numpy as np
import analyse_all_followers as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firebird-reference',type=Path,required=True)
    args=parser.parse_args()
    factory=study.published.make_matchmaker_follower
    study.retain_arzt_array();owned=study.published.make_matchmaker_follower
    rng=np.random.default_rng(187)
    ref=rng.uniform(size=(12,700));ref/=np.linalg.norm(ref,axis=0)
    live=ref[:,np.minimum(np.arange(500)*1.1,699).astype(int)]
    np.testing.assert_array_equal(study.decode_cc(ref,live,prefix=False),study.decode_cc(ref,live))
    checks=[]
    for row in json.loads(study.INDEX.read_text())['rows']:
        result,t,_=study.load_case(row)
        ref,live,meta=study.features(row['pair'],result,args)
        short=row['pair'] in ('firebird','chopin_black_keys')
        cc='concertcue' in row['method']
        if not short:live=live[:,:600] if cc else live[:600]
        if cc:
            x=study.decode_cc(ref,live)
            if short:np.testing.assert_array_equal(x,study.decode_cc(ref,live,prefix=False))
        else:
            x,*_=study.published.decode_matchmaker(ref,live,method=row['method'],frame_rate=30,progress=False)
            if short:
                study.published.make_matchmaker_follower=factory
                y,*_=study.published.decode_matchmaker(ref,live,method=row['method'],frame_rate=30,progress=False)
                study.published.make_matchmaker_follower=owned
                np.testing.assert_array_equal(x,y)
        original=t['online_ref_t'][:len(x)]-result['reference']['shared_span']['ref_start']
        count=int(np.sum(x!=original));maximum=float(np.max(np.abs(x-original)))
        if row['pair']=='firebird' and 'arzt' in row['method']:
            assert count==2 and maximum<1/30+1e-6
        else:np.testing.assert_array_equal(x,original)
        checks.append({'pair':row['pair'],'method':row['method'],'frames':len(x),
                'full_short_recording':short,'original_trace_differing_frames':count,
                'maximum_difference_s':maximum,'wrapper_parity_checked':short})
        print(checks[-1],flush=True)
    study.write(study.OUT/'decoder_validation.json',{'checks':checks,
        'synthetic_concertcue_allocation_parity_500_frames':True,
        'script':study.record(Path(__file__).resolve()),'provenance':study.provenance()})


if __name__=='__main__':main()
