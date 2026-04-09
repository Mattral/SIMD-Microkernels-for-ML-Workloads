# Benchmark Results

This folder stores JSON benchmark outputs produced by the `benchmarks/run_bench.sh` script.

## Structure

- `bench_results.json` — the latest benchmark run output.

The JSON output includes an array of benchmark records with fields such as:
- `name`
- `category`
- `M`, `N`, `K`
- `n`
- `min_cycles`
- `median_cycles`
- `gflops`

Use this file for automated performance comparisons and reporting.
