#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
results_dir="$repo_root/benchmarks/results"
results_file="$results_dir/bench_results.json"

mkdir -p "$build_dir"
mkdir -p "$results_dir"
cd "$build_dir"

cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMD_ML_OPENMP=ON
cmake --build . --target bench

printf "Running benchmark and writing JSON results to %s\n" "$results_file"
./bench --json "$results_file" --warmup 3 --reps 10

printf "Benchmark complete. JSON results available at %s\n" "$results_file"
