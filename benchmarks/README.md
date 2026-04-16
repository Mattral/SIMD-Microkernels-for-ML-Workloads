# Benchmarks

This directory contains tooling for running the `bench` benchmark harness and
saving structured results.

## Run the benchmark

```bash
./benchmarks/run_bench.sh
```

This script will:

- configure `build/` with `Release` and OpenMP enabled
- build the `bench` executable
- execute `./bench --json benchmarks/results/bench_results.json`

## Output

- `benchmarks/results/bench_results.json` stores the benchmark results in JSON.
- The benchmark harness also prints a human-readable summary to stdout.

## Notes

The benchmark harness uses `RDTSC` with `LFENCE`/`RDTSCP` serialisation. Results
are intended for comparative and reproducibility analysis, not as a calibrated
performance certification suitable for production benchmarking.
