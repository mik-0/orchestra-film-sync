"""Compare a corrected dense reference with reviewed and older hand anchors.

The output is an ``annotate.py``-compatible JSON containing only timestamps
whose disagreement exceeds the requested tolerance.  This keeps follow-up
listening focused on actual conflicts rather than making the reviewer repeat an
entire recording.
"""

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


import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--manual", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_anchors(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["anchors"]


def compare(
    *,
    dense_live: np.ndarray,
    dense_ref: np.ndarray,
    anchors: list[dict],
    source: str,
    tolerance: float,
) -> list[dict]:
    flagged = []
    for index, anchor in enumerate(anchors):
        live_t = float(anchor["live_t"])
        given_ref = float(anchor["ref_t"])
        if live_t < dense_live[0] or live_t > dense_live[-1]:
            continue
        corrected_ref = float(np.interp(live_t, dense_live, dense_ref))
        error = corrected_ref - given_ref
        if abs(error) <= tolerance:
            continue
        flagged.append({
            "live_t": round(live_t, 4),
            "ref_t": round(corrected_ref, 4),
            "comparison_source": source,
            "comparison_index": index + 1,
            "comparison_ref_t": round(given_ref, 4),
            "difference_s": round(error, 4),
        })
    return flagged


def main() -> None:
    args = parse_args()
    with open(args.reference, encoding="utf-8") as handle:
        reference = json.load(handle)
    dense_path = args.reference.parent / reference["dense_path"]
    dense = np.load(dense_path)
    dense_live = dense["live_t"]
    dense_ref = dense["ref_t"]

    comparisons = [("reviewed", args.reviewed)]
    if args.manual is not None:
        comparisons.append(("manual", args.manual))

    flags = []
    summaries = []
    for label, path in comparisons:
        anchors = load_anchors(path)
        current = compare(
            dense_live=dense_live,
            dense_ref=dense_ref,
            anchors=anchors,
            source=label,
            tolerance=args.tolerance,
        )
        flags.extend(current)
        summaries.append({
            "source": label,
            "path": str(path),
            "anchor_count": len(anchors),
            "flag_count": len(current),
        })

    # Keep separate evidence when two sources flag the same time, but collapse
    # the listening anchors so annotate.py presents that position only once.
    unique = {}
    for flag in flags:
        key = round(float(flag["live_t"]), 3)
        unique.setdefault(key, {
            "live_t": flag["live_t"],
            "ref_t": flag["ref_t"],
            "reasons": [],
        })["reasons"].append({
            key: value for key, value in flag.items()
            if key not in ("live_t", "ref_t")
        })
    review_anchors = sorted(unique.values(), key=lambda item: item["live_t"])

    payload = {
        "live_file": reference["live_file"],
        "ref_file": reference["ref_file"],
        "sr": reference["sr"],
        "source": "alignment_reference_conflict_review",
        "reference": str(args.reference),
        "tolerance_s": args.tolerance,
        "comparisons": summaries,
        "anchors": review_anchors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(args.output)
    print(f"{args.reference.name}: {len(review_anchors)} review positions")
    for anchor in review_anchors:
        print(
            f"  live={anchor['live_t']:.3f}s -> ref={anchor['ref_t']:.3f}s "
            f"({len(anchor['reasons'])} conflict)"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
