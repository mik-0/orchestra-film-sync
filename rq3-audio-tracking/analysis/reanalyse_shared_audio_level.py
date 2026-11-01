"""Use one recorded RMS series per pair for every follower's level analysis.

The original ConcertCue runner computes RMS from decoded arrays; Matchmaker's
runner reads short source-file windows. Keep those results, and remove the small
loader differences from the cross-method comparison using the Dixon RMS series.
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

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import analyse_all_followers as study


def main():
    rows=json.loads(study.INDEX.read_text())['rows']
    for row in rows:
        if 'concertcue' not in row['method']:continue
        pair=row['pair'];target=study.OUT/f'{pair}_concertcue_shared_energy.json'
        if target.exists():continue
        result,trace,inputs=study.load_case(row)
        dixon=next(r for r in rows if r['pair']==pair and 'dixon' in r['method'])
        _,shared,shared_inputs=study.load_case(dixon)
        np.testing.assert_array_equal(trace['sample_t'],shared['sample_t'])
        old=json.loads((study.OUT/f'{pair}_concertcue_traces.json').read_text())
        new=json.loads(json.dumps(old));energy=shared['sample_energy_ratio']
        new['inputs'].append(shared_inputs[1])
        new['shared_audio_level']={'source':shared_inputs[1],
             'maximum_difference_from_original_ratio':float(np.max(np.abs(energy-trace['sample_energy_ratio'])))}
        new['correlations']=[]
        for block in study.BLOCKS:
            b=study.moving_block_rank_correlation(np.abs(trace['sample_error_s']),energy,
                    block_size=block,repetitions=10000,seed=study.SEED)
            new['correlations'].append({'block_seconds':block,**asdict(b)})
            print(pair,block,b.lower,b.upper,flush=True)
        for f in new['failure_sensitivity']:
            f['quiet_counts']=[{'ratio':q,**study.low_energy_counts(f['episodes'],trace['sample_t'],energy,q)}
                               for q in study.QUIET]
        new['provenance']=study.provenance()
        new['provenance']['shared_energy_script']=study.record(Path(__file__).resolve())
        study.write(target,new)


if __name__=='__main__':main()
