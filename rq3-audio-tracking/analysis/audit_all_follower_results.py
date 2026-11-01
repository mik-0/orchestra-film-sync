"""Check saved analysis arithmetic, provenance, timing, and restart coverage."""

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
from scipy.stats import spearmanr
import analyse_all_followers as study


def main():
    evidence=[]
    index=json.loads(study.INDEX.read_text())['rows']
    for row in index:
        pair=row['pair'];method=next(m for m in ['arzt','dixon','concertcue'] if m in row['method'])
        key=f'{pair}_{method}'
        result,t,inputs=study.load_case(row)
        p=study.OUT/f'{key}_shared_energy.json'
        if not p.exists():p=study.OUT/f'{key}_traces.json'
        d=json.loads(p.read_text())
        energy=t['sample_energy_ratio']
        if method=='concertcue':
            assert 'shared_audio_level' in d
            with np.load(study.artifact_path(d['shared_audio_level']['source']['path'])) as z:
                energy=z['sample_energy_ratio']
        for item in d['inputs']:
            assert study.published.sha256(study.artifact_path(item['path']))==item['sha256']
        rho=float(spearmanr(np.abs(t['sample_error_s']),energy).statistic)
        for c in d['correlations']:
            np.testing.assert_allclose(c['rho'],rho,atol=1e-14)
            assert c['defined_resamples']+c['undefined_resamples']==10000
        for f in d['failure_sensitivity']:
            eps=study.failure_episodes(t['online_live_t'],t['error_s'],threshold_s=f['error_limit_s'],minimum_duration_s=1.)
            assert eps==f['episodes']
            for q in f['quiet_counts']:
                expected=study.low_energy_counts(eps,t['sample_t'],energy,q['ratio'])
                assert all(q[k]==v for k,v in expected.items())
        r=json.loads((study.OUT/f'{key}_restarts.json').read_text())
        assert len(r['rows'])==len(d['bouts'])*len(study.LEADS)
        expected_keys={(i,lead) for i in range(len(d['bouts'])) for lead in study.LEADS}
        assert {(x['bout'],x['lead_s']) for x in r['rows']}==expected_keys
        for x in r['rows']:
            if not x['available']:
                assert x['requested_start_s']<t['online_live_t'][0]
                continue
            start,end=x['live_start_frame'],x['live_end_frame']
            assert x['actual_start_s']>=x['requested_start_s']-1e-8
            assert x['target_s']==t['online_live_t'][end]
            assert x['processed_frames']==end-start+1
            assert x['target_reference_s']==t['ground_truth_ref_t'][end]
            expected_clock=x['supplied_reference_s']+x['target_s']-x['actual_start_s']
            assert x['clock_reference_s']==expected_clock
            assert x['restart_abs_error_s']==abs(x['estimate_reference_s']-x['target_reference_s'])
            assert x['clock_abs_error_s']==abs(expected_clock-x['target_reference_s'])
            ref_step=4096/44100 if method=='concertcue' else 1/30
            assert abs(x['oracle_quantisation_error_s'])<=ref_step/2+1e-8
            assert x['actual_lead_s']<=x['lead_s']+1e-8
        assert r['summary']==study.summarize(r['rows'])
        evidence.append({'pair':pair,'method':method,'trace_metrics_and_correlations_checked':True,
            'restart_trials':len(r['rows']),'available':sum(x['available'] for x in r['rows']),
            'restart_coverage_timing_clock_and_errors_checked':True})
    study.write(study.OUT/'result_audit.json',{'cases':evidence,'passed':True,
                'audit_script':study.record(Path(__file__).resolve())})
    print('All twelve cases passed:',sum(e['restart_trials'] for e in evidence),'restart rows.')


if __name__=='__main__':main()
