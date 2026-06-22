# Benchmark Results

This directory stores JSON outputs from `bench_stat` (statistical harness)
and `bench` (cycle-accurate RDTSC harness).

---

## Files

| File | Produced by | Contents |
|---|---|---|
| `gemm_results.json` | `bench_stat` — **committed baseline** | Controlled reference run for regression gating |
| `bench_stat.json` | `bench_stat` — transient CI output | Latest CI run (not committed; compared against baseline) |
| `bench_rdtsc.json` | `bench` — transient | Cycle-accurate RDTSC results |

---

## `bench_stat` JSON schema

```json
{
  "gemm": [
    {
      "kernel":        "simd_packed",
      "N":             256,
      "min_ms":        1.864,
      "mean_ms":       1.891,
      "median_ms":     1.877,
      "stddev_ms":     0.012,
      "ci95_half_ms":  0.015,
      "gflops_peak":   18.01,
      "gflops_mean":   17.76
    }
  ],
  "gelu": [
    {
      "kernel":        "gelu_avx2",
      "n":             1048576,
      "min_ms":        0.902,
      "mean_ms":       0.956,
      "median_ms":     0.973,
      "stddev_ms":     0.053,
      "ci95_half_ms":  0.068,
      "gelems_per_sec": 1.162
    }
  ]
}
```

---

## `bench` (RDTSC) JSON schema

```json
{
  "benchmarks": [
    {
      "name":          "SIMD   GEMM  256x256",
      "category":      "gemm",
      "M": 256, "N": 256, "K": 256,
      "n": 0,
      "min_cycles":    4213936,
      "median_cycles": 4250100,
      "gflops":        16.72
    }
  ]
}
```

---

## Establishing a new baseline

Run `bench_stat` on dedicated hardware with the CPU frequency locked:

```bash
sudo cpupower frequency-set -g performance
taskset -c 0 ./build/bench_stat \
    --reps 50 \
    --sizes 64,128,256,512,1024 \
    --output benchmarks/results/gemm_results.json
```

Commit `gemm_results.json` to track regressions in CI via
`benchmarks/check_regression.py`.
