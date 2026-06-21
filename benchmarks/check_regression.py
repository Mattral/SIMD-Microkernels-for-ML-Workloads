#!/usr/bin/env python3
"""
check_regression.py — CI Regression Gating for IntrinsicML Benchmarks

Compares a current benchmark JSON result against a committed baseline.
Fails (exit code 1) if any kernel's mean GFLOPS dropped by more than
--max-regression-pct percent versus the baseline.

Usage:
    python benchmarks/check_regression.py \\
        --baseline benchmarks/results/gemm_results.json \\
        --current  benchmarks/results/ci_gemm_results.json \\
        --max-regression-pct 15

JSON schema (matches bench_stat.cpp output):
{
  "gemm": [
    {"kernel": "simd_packed", "N": 256, "gflops_mean": 18.5, ...},
    ...
  ],
  "gelu": [
    {"kernel": "gelu_avx2", "n": 65536, "gelems_per_sec": 12.3, ...},
    ...
  ]
}
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def build_key(entry: dict, kind: str) -> str:
    """Build a unique key for a benchmark entry."""
    if kind == "gemm":
        return f"{entry['kernel']}@N={entry['N']}"
    elif kind == "gelu":
        return f"{entry['kernel']}@n={entry['n']}"
    return str(entry)


def check_regressions(baseline: dict, current: dict,
                      max_regression_pct: float) -> int:
    """
    Returns the number of regressions detected.
    Prints a summary table.
    """
    regressions = 0
    total_checks = 0

    print(f"\n{'Key':<40} {'Baseline':>12} {'Current':>12} {'Δ%':>8} {'Status':>10}")
    print("-" * 90)

    for kind in ("gemm", "gelu"):
        if kind not in baseline or kind not in current:
            continue

        perf_key = "gflops_mean" if kind == "gemm" else "gelems_per_sec"
        label    = "GFLOPS" if kind == "gemm" else "GElems/s"

        # Index baseline entries by key
        base_map = {}
        for entry in baseline[kind]:
            k = build_key(entry, kind)
            base_map[k] = entry.get(perf_key, 0.0)

        for entry in current[kind]:
            k = build_key(entry, kind)
            cur_val = entry.get(perf_key, 0.0)

            if k not in base_map:
                print(f"  {k:<40} {'N/A':>12} {cur_val:>12.2f}  {'N/A':>8}  NEW")
                continue

            base_val = base_map[k]
            if base_val <= 0:
                continue

            pct_change = 100.0 * (cur_val - base_val) / base_val
            total_checks += 1

            status = "OK"
            if pct_change < -max_regression_pct:
                status = "REGRESSION"
                regressions += 1

            print(f"  {k:<40} {base_val:>12.2f} {cur_val:>12.2f} {pct_change:>+8.1f}%  {status}")

    print()
    print(f"Checked {total_checks} entries | Regressions: {regressions} "
          f"| Threshold: -{max_regression_pct:.0f}%")

    return regressions


def main():
    parser = argparse.ArgumentParser(description="CI regression checker for IntrinsicML benchmarks")
    parser.add_argument("--baseline", required=True,
                        help="Path to committed baseline JSON file")
    parser.add_argument("--current", required=True,
                        help="Path to current CI run JSON file")
    parser.add_argument("--max-regression-pct", type=float, default=15.0,
                        help="Maximum allowed GFLOPS/throughput drop in %% (default: 15)")
    args = parser.parse_args()

    print(f"IntrinsicML Benchmark Regression Check")
    print(f"  Baseline : {args.baseline}")
    print(f"  Current  : {args.current}")
    print(f"  Threshold: -{args.max_regression_pct:.0f}% maximum drop")

    baseline = load_json(args.baseline)
    current  = load_json(args.current)

    regressions = check_regressions(baseline, current, args.max_regression_pct)

    if regressions > 0:
        print(f"\n❌  FAIL: {regressions} performance regression(s) detected.")
        print("   Investigate before merging. Consider:")
        print("   1. Noise? Re-run bench_stat with --reps 50 on a dedicated machine.")
        print("   2. Real regression? Profile with perf stat or VTune.")
        sys.exit(1)
    else:
        print(f"\n✅  PASS: No regressions detected (threshold: -{args.max_regression_pct:.0f}%)")
        sys.exit(0)


if __name__ == "__main__":
    main()
