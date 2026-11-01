"""Summarise the all-follower study and render figures from its saved traces."""

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

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyse_all_followers as study

PAIRS=['firebird','chopin_black_keys','yuja_fedorova','daphnis_chloe']
LABELS=['Firebird','Chopin','Rachmaninoff','Daphnis et Chloé']
METHODS=['arzt','dixon','concertcue']
NAMES=['Arzt','Dixon','ConcertCue']
COLORS=['#287aa6','#c76b19','#7b4992','#28836d']


def main():
    records={}
    aggregate={}
    for method in METHODS:
        ds=[]
        for pair in PAIRS:
            p=study.OUT/f'{pair}_{method}_shared_energy.json'
            if not p.exists():p=study.OUT/f'{pair}_{method}_traces.json'
            ds.append(json.loads(p.read_text()))
        for pair,d in zip(PAIRS,ds):records[pair,method]=d
        senses=[]
        for i,limit in enumerate(study.LIMITS):
            fs=[d['failure_sensitivity'][i] for d in ds]
            episodes=[e for f in fs for e in f['episodes']]
            senses.append({'error_limit_s':limit,'count':len(episodes),
                'behind':sum(f['peak_behind_count'] for f in fs),
                'duration':study.episode_duration_summary(episodes),
                'quiet':[{'ratio':q,**{k:sum(f['quiet_counts'][j][k] for f in fs)
                  for k in ['episode_count','episodes_without_preceding_sample',
                            'episodes_with_preceding_sample','episodes_after_low_energy']}}
                  for j,q in enumerate(study.QUIET)]})
        restarts=[]
        restart_by_pair={}
        for pair in PAIRS:
            p=study.OUT/f'{pair}_{method}_restarts.json'
            if p.exists():
                r=json.loads(p.read_text());restarts.extend(r['rows']);restart_by_pair[pair]=r['summary']
        aggregate[method]={'failure_sensitivity':senses,'merged_bouts':sum(len(d['bouts']) for d in ds),
              'restart_complete':len(restart_by_pair)==4,
              'restart_summary':study.summarize(restarts),'restart_by_pair':restart_by_pair}
    study.write(study.OUT/'summary.json',{'aggregate':aggregate,
        'records':{f'{pair}_{method}':d for (pair,method),d in records.items()},
        'provenance':{'summary_script':study.record(Path(__file__).resolve()),
                      'inputs':[study.record(p) for p in sorted(study.OUT.glob('*_traces.json'))]+[
                          study.record(p) for p in sorted(study.OUT.glob('*_shared_energy.json'))]+[
                          study.record(p) for p in sorted(study.OUT.glob('*_restarts.json'))]}})
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':11,'axes.titlesize':12,
                         'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10})
    image_dir=study.ROOT/'results/figures'
    image_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,3,figsize=(8.3,8),layout='constrained')
    for i,pair in enumerate(PAIRS):
        for j,method in enumerate(METHODS):
            d=records[pair,method]
            trace=study.artifact_path(d['inputs'][1]['path'])
            with np.load(trace) as z:
                t=z['sample_t']/60;e=z['sample_error_s']
            ax=axes[i,j];ax.plot(t,e,color=COLORS[i],lw=.85)
            ax.axhspan(-1,1,color='#a8cbb0',alpha=.4,zorder=-1)
            ax.axhline(0,color='#888888',lw=.5)
            ax.set_title(f'{LABELS[i]} · {NAMES[j]}',fontsize=10)
            ax.set_xlim(t[0],t[-1]);ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Position error (s)')
            if i==3:ax.set_xlabel('Live time (min)')
    fig.savefig(image_dir/'audio_failure_all_methods_map.png',dpi=240)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9.2,7.2),layout='constrained')
    for i,method in enumerate(METHODS):
        for j,pair in enumerate(PAIRS):
            d=records[pair,method]
            with np.load(study.ROOT/d['inputs'][1]['path']) as z:
                errors=np.abs(z['error_s'])
            limits=np.array([.05,.1,.5,1,2])
            axes[i,0].plot(limits,[100*np.mean(errors<=x) for x in limits],'-o',ms=3,color=COLORS[j],label=LABELS[j])
            fs=d['failure_sensitivity']
            axes[i,1].plot(study.LIMITS,[100*f['time_fraction'] for f in fs],'-o',ms=3,color=COLORS[j])
            axes[i,2].plot(study.LIMITS,[100*f['peak_behind_count']/f['count'] if f['count'] else np.nan for f in fs],'-o',ms=3,color=COLORS[j])
        axes[i,0].set_xscale('log');axes[i,0].set_xticks([.05,.1,.5,1,2],['.05','.1','.5','1','2'])
        axes[i,0].set_ylabel(NAMES[i]+'\nPercentage (%)')
        for j in range(3):
            ax=axes[i,j];ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if i==2:ax.set_xlabel('Error limit (s)')
            if j:ax.set_xticks(study.LIMITS)
    for ax,title in zip(axes[0],['Estimates within limit','Time in sustained failure','Failure peaks behind']):ax.set_title(title)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(image_dir/'audio_failure_all_methods_sensitivity.png',dpi=240)
    plt.close(fig)
    print('Summary and two figures written. Restart completion:',{m:a['restart_complete'] for m,a in aggregate.items()})


if __name__=='__main__':main()
