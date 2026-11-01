#!/usr/bin/env python3
"""Generate Figure 6.4 from the preserved session; omit the two invalid targets.

Run: python rq1-acceptable-timing/analysis/plot_session_heatmap.py
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
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = _REPO_ROOT / "rq1-acceptable-timing/figures/acceptability_session.png"
EXCLUDED = {'138->139', '139->140'}


def main():
    rows = json.loads((HERE / 'sessie_20260630-125252-9crr_alle-overgangen.json').read_text())
    usable = [r for r in rows if r['transition'] not in EXCLUDED]
    targets = sorted({r['transition'] for r in usable}, key=lambda t: int(t.split('->')[1]))
    offsets = sorted({float(r['offset_s']) for r in usable})
    if (len(rows), len(usable), len(targets), len(offsets)) != (105, 91, 13, 7):
        raise ValueError('Unexpected session dimensions')
    ratings = {}
    for row in usable:
        key = (row['transition'], float(row['offset_s']))
        if key in ratings or row['rating_value'] not in (-1, 1):
            raise ValueError('Duplicate or non-binary rating')
        ratings[key] = int(row['rating_value'] == 1)
    matrix = np.array([[ratings[t, o] for o in offsets] for t in targets])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    colors = ['#e7eaee', '#286582']
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    labels = [t.replace('start', 'Start').replace('->', ' → ') for t in targets]
    ax.set_yticks(range(len(targets)), labels)
    ax.set_xticks(range(len(offsets)), [f'{v:+.2f}' if v else '0' for v in offsets])
    ax.set_xlabel('Tested cut offset (s)\nNegative = early; positive = late', labelpad=12)
    ax.set_ylabel('Transition', labelpad=12)
    ax.set_xticks(np.arange(-0.5, len(offsets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='both', length=0, pad=8)
    ax.spines[:].set_visible(False)
    ax.legend(handles=[Patch(facecolor=colors[1], label='Acceptable'),
                       Patch(facecolor=colors[0], label='Not acceptable')],
              loc='lower center', bbox_to_anchor=(0.5, 1.025), ncol=2,
              frameon=False, handlelength=1.3, columnspacing=2.0)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.18)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUTPUT}: 13 transitions, 7 offsets, 91 unchanged ratings')


if __name__ == '__main__':
    main()
