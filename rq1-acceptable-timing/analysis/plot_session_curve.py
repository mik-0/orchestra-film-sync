#!/usr/bin/env python3
"""Regenerate the pooled acceptability curve from the preserved session."""

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


from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = _REPO_ROOT
INPUT = HERE / "sessie_20260630-125252-9crr_alle-overgangen.json"
OUTPUT = REPOSITORY_ROOT / "results/figures/acceptability_session_curve.png"
EXCLUDED = {"138->139", "139->140"}


def crossing(x0: float, y0: float, x1: float, y1: float) -> float:
    """Return the x coordinate at which the line between two points reaches zero."""

    return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    usable = [row for row in rows if row["transition"] not in EXCLUDED]
    transitions = sorted({row["transition"] for row in usable})
    if len(rows) != 105 or len(usable) != 91 or len(transitions) != 13:
        raise RuntimeError(
            "unexpected acceptability study session size: "
            f"all={len(rows)}, usable={len(usable)}, transitions={len(transitions)}"
        )

    by_offset: dict[float, list[float]] = defaultdict(list)
    for row in usable:
        by_offset[float(row["offset_s"])].append(float(row["rating_value"]))

    offsets = np.array(sorted(by_offset))
    means = np.array([np.mean(by_offset[offset]) for offset in offsets])
    early = crossing(offsets[0], means[0], offsets[1], means[1])
    late = crossing(offsets[-3], means[-3], offsets[-2], means[-2])

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.axvspan(early, late, color="#dcefd8", alpha=0.9, label="pooled acceptable window")
    ax.axhline(0, color="#b7b7b7", linewidth=0.8)
    ax.axvline(0, color="#d5d5d5", linewidth=0.8, linestyle="--")
    ax.plot(offsets, means, color="#1764aa", marker="o", linewidth=2.1, markersize=5)

    peak_index = int(np.argmax(means))
    ax.scatter(
        [offsets[peak_index]],
        [means[peak_index]],
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.8,
        s=75,
        zorder=3,
    )
    ax.annotate(
        f"peak {means[peak_index]:+.2f}",
        (offsets[peak_index], means[peak_index]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color="#d62728",
        fontsize=9,
    )
    ax.annotate(f"{early:.2f} s", (early, 0), xytext=(-7, -14), textcoords="offset points", ha="right", fontsize=9)
    ax.annotate(f"+{late:.2f} s", (late, 0), xytext=(7, -14), textcoords="offset points", ha="left", fontsize=9)

    ax.set_title("Acceptability by firing offset - 13 usable transitions, 1 judgement each", fontsize=11)
    ax.set_xlabel("cut firing offset (s)    -    negative = cut early, positive = cut late")
    ax.set_ylabel("mean rating\n(acceptable +1 / not acceptable -1)")
    ax.set_xticks(offsets)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
