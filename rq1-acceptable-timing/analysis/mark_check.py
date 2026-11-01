#!/usr/bin/env python3
"""#36: sanity-check the 16 transition marks against audio onset novelty.
A cut lands on a musical moment, so each mark should sit near a strong onset.
Reports signed distance mark -> nearest strong onset; big |offset| = suspect mark."""

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

import numpy as np, json
import os
from pathlib import Path
import librosa

AUDIO_CACHE = os.environ.get("FIREBIRD_AUDIO_CACHE")
if AUDIO_CACHE is None and os.environ.get("FIREBIRD_VIDEO"):
    AUDIO_CACHE = f"{os.environ['FIREBIRD_VIDEO']}.22050.npz"
if AUDIO_CACHE is None:
    raise SystemExit("set FIREBIRD_AUDIO_CACHE to the cached conductor-audio .npz file")

d = np.load(Path(AUDIO_CACHE).expanduser())
y, sr = d["y"].astype(np.float32), int(d["sr"])
hop = 256
env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
times = librosa.times_like(env, sr=sr, hop_length=hop)
# strong onsets only: peaks above 75th percentile of env
onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="time",
                                    delta=np.percentile(env, 75)/env.max(), normalize=True)
print(f"{len(onsets)} strong onsets over {len(y)/sr:.0f}s")

ROOT = _REPO_ROOT
marks = []
for ln in (ROOT/"rq2-conductor-cues/features/firebird_timeline.txt").read_text().splitlines():
    s = ln.strip()
    if not s or s.startswith("#"): continue
    seg, ts = s.split()
    mm, ss, ff = map(int, ts.split(":"))
    marks.append((seg, mm*60+ss+ff/25))

print(f"{'seg':>4} {'mark':>7} {'nearest onset':>13} {'signed off':>10}   (negative = onset BEFORE mark = mark late)")
for seg, t in marks:
    i = np.argmin(np.abs(onsets - t))
    off = onsets[i] - t
    flag = "  <-- SUSPECT" if abs(off) > 0.35 else ""
    star = " *" if seg in ("139","140","142") else ""
    print(f"{seg:>4} {t:>7.2f} {onsets[i]:>13.2f} {off:>+10.2f}{flag}{star}")

# For the three named suspects: show the 3 strongest novelty peaks within +-2s
print("\nstrongest novelty peaks within +-2.0s of the suspect marks:")
for seg, t in marks:
    if seg not in ("139","140","142"): continue
    m = (times > t-2.0) & (times < t+2.0)
    tt, ee = times[m], env[m]
    top = tt[np.argsort(ee)[-5:]]
    print(f"  {seg} (mark {t:.2f}): peaks at " + ", ".join(f"{x:.2f}" for x in sorted(top)))
