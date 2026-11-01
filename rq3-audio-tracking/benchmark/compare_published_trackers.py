"""Create a source-verified descriptive table for published tracker results."""

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
import csv
import hashlib
import json
from pathlib import Path


PAIR_ORDER = [
    "firebird",
    "chopin_black_keys",
    "yuja_fedorova",
    "daphnis_chloe",
]
METHOD_ORDER = [
    "external_matchmaker_arzt_v0_3_0",
    "external_matchmaker_dixon_v0_3_0",
    "external_concertcue_otw_2876aaca",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_method(result: dict) -> str:
    method = result["method"]
    if method == "hmm_uninterrupted":
        return "internal_hmm_uninterrupted"
    return method


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_pairs = {item["pair"]: item for item in protocol["recording_pairs"]}
    protocol_methods = {
        item["method"]
        for item in protocol["methods"]
        if item["role"] == "published_external_baseline"
    }
    if protocol_methods != set(METHOD_ORDER):
        raise SystemExit("protocol method set does not match summary implementation")

    results: dict[tuple[str, str], tuple[Path, dict]] = {}
    runtime_failures: dict[tuple[str, str], tuple[Path, dict]] = {}
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        pair = result["pair"]
        method = normalized_method(result)
        if pair not in protocol_pairs:
            raise SystemExit(f"{pair}: absent from protocol")
        if method not in protocol_methods:
            raise SystemExit(f"{pair}/{method}: absent from protocol")
        expected = protocol_pairs[pair]
        if result["reference"]["sha256"] != expected["reference_sha256"]:
            raise SystemExit(f"{pair}/{method}: reference hash mismatch")
        if result["reference"]["dense_sha256"] != expected["dense_sha256"]:
            raise SystemExit(f"{pair}/{method}: dense reference hash mismatch")
        key = (pair, method)
        if key in results or key in runtime_failures:
            raise SystemExit(f"duplicate result: {pair}/{method}")
        if result["status"] == "runtime_failure":
            runtime_failures[key] = (path, result)
        else:
            results[key] = (path, result)

    rows = []
    coverage = []
    for pair in PAIR_ORDER:
        for method in METHOD_ORDER:
            key = (pair, method)
            if key in runtime_failures:
                path, failure = runtime_failures[key]
                coverage.append(
                    {
                        "pair": pair,
                        "method": method,
                        "status": "runtime_failure",
                        "result_path": str(path),
                        "exit_code": failure["runtime_failure"]["exit_code"],
                    }
                )
                continue
            if key not in results:
                if (
                    method == "external_concertcue_otw_2876aaca"
                    and pair in {"yuja_fedorova", "daphnis_chloe"}
                ):
                    reason = "not_run_exact_upstream_implementation_exceeds_1_GiB"
                else:
                    reason = "missing_result"
                coverage.append({"pair": pair, "method": method, "status": reason})
                continue
            path, result = results[key]
            metrics = result["metrics"]
            failure = result["failure_summary"]
            provenance = result["provenance"]
            runtime = provenance.get(
                "total_runtime_seconds", provenance.get("runtime_seconds")
            )
            rows.append(
                {
                    "pair": pair,
                    "method": method,
                    "median_abs_error_s": metrics["median_abs_error_s"],
                    "p90_abs_error_s": metrics["p90_abs_error_s"],
                    "max_abs_error_s": metrics["max_abs_error_s"],
                    "rate_within_1s": metrics["rate_within_1s"],
                    "failure_episode_count": failure["episode_count"],
                    "failure_time_fraction": failure["failure_time_fraction"],
                    "runtime_s": runtime,
                    "result_path": str(path),
                }
            )
            coverage.append({"pair": pair, "method": method, "status": "completed"})

    unexpected_missing = [item for item in coverage if item["status"] == "missing_result"]
    if unexpected_missing:
        missing = [f"{item['pair']}/{item['method']}" for item in unexpected_missing]
        raise SystemExit(f"missing required results: {missing}")

    payload = {
        "status": "final" if all(
            result[1]["status"] == "final" for result in results.values()
        ) else "provisional",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "reporting_unit": "recording pair",
        "inferential_claim": False,
        "ranking_reported": False,
        "coverage": coverage,
        "runtime_failures": [
            {"result_path": str(path), **result}
            for path, result in runtime_failures.values()
        ],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(args.output_json)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
