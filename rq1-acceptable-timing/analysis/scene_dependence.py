#!/usr/bin/env python3
"""#37: does the per-transition tolerance window relate to the transition's character?
Tolerance breadth (n offsets accepted, of 7) vs: cue label (8v8 split), pre-cut energy,
energy jump at the cut, tracker |err|. n=13 usable transitions, 1 repeat -> exploratory.
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

import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = _REPO_ROOT
BAD = {"138->139", "139->140"}

RESULTS_PATH = Path(os.environ.get(
    "ACCEPTABILITY_RESULTS",
    Path(__file__).parent / "sessie_20260630-125252-9crr_alle-overgangen.json",
)).expanduser()
AUDIO_CACHE = os.environ.get("FIREBIRD_AUDIO_CACHE")
if AUDIO_CACHE is None and os.environ.get("FIREBIRD_VIDEO"):
    AUDIO_CACHE = f"{os.environ['FIREBIRD_VIDEO']}.22050.npz"
if AUDIO_CACHE is None:
    raise SystemExit("set FIREBIRD_AUDIO_CACHE to the cached conductor-audio .npz file")

rows = json.load(open(RESULTS_PATH))
av2 = json.loads((ROOT/"rq2-conductor-cues/features/analysis_results/analysis_v2.json").read_text())
cue_label = {t["seg_id"]: l for t, l in zip(av2["transitions"], av2["labels"])}
t_time = {t["seg_id"]: t["time"] for t in av2["transitions"]}

# audio energy (conductor/live)
d = np.load(Path(AUDIO_CACHE).expanduser())
y, sr = d["y"].astype(float), int(d["sr"])
def rms(t0, t1):
    a, b = max(0, int(t0*sr)), min(len(y), int(t1*sr))
    return float(np.sqrt(np.mean(y[a:b]**2))) if b > a else 0.0
mids = np.arange(0.2, len(y)/sr-0.2, 0.2)
med = float(np.median([rms(m-0.2, m+0.2) for m in mids]))

# tracker error at transitions
ae = json.loads((ROOT/"shared/audio/output/firebird_autoeval.json").read_text())
pa = [p for p in ae["per_anchor"] if p.get("online_err") is not None]
lt = np.array([p["live_t"] for p in pa]); er = np.abs([p["online_err"] for p in pa])
o = np.argsort(lt); lt, er = lt[o], np.array(er)[o]

per = {}
for r in rows:
    tr = r["transition"]
    if tr in BAD: continue
    per.setdefault(tr, {})[round(float(r["offset_s"]), 3)] = 1 if r["rating"] == "acceptabel" else 0

print(f"{'transition':>12} {'acc/7':>6} {'cued':>5} {'pre/med':>8} {'jump':>6} {'|err|':>6}")
out = []
for tr, offs in per.items():
    inc = tr.split("->")[1]
    T = t_time[inc]
    n_acc = sum(offs.values())
    pre = rms(T-1.0, T-0.15)/med
    post = rms(T+0.05, T+1.0)/med
    jump = post/pre if pre > 0 else np.nan
    terr = float(np.interp(T, lt, er))
    cued = cue_label[inc]
    out.append(dict(tr=tr, n=n_acc, cued=cued, pre=pre, jump=jump, terr=terr))
    print(f"{tr:>12} {n_acc:>6} {cued:>5} {pre:>8.2f} {jump:>6.2f} {terr:>6.2f}")

n = np.array([x["n"] for x in out], float)
for name, v in [("pre-cut energy", [x["pre"] for x in out]),
                ("energy jump", [x["jump"] for x in out]),
                ("tracker |err|", [x["terr"] for x in out])]:
    rho, p = spearmanr(n, v)
    print(f"Spearman(acceptance breadth, {name}) = {rho:+.2f} (p={p:.2f})")
c = n[[x["cued"] == 1 for x in out]]; nc = n[[x["cued"] == 0 for x in out]]
u, p = mannwhitneyu(c, nc, alternative="two-sided")
print(f"cued (n={len(c)}): median breadth {np.median(c):.1f} | non-cue (n={len(nc)}): {np.median(nc):.1f}  (Mann-Whitney p={p:.2f})")
