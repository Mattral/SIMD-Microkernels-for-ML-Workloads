# BENCHMARKS

This file contains verbatim output captured from running `./bench` on the development host. It is intended as a reproducible reference for measured per-core GFLOPS and roofline-summary data.

Captured output (run date: 2026-05-30):

```
╔══════════════════════════════════════════════════╗
║  SIMD-ML-Microkernels  ·  Cycle-Accurate Bench  ║
╚══════════════════════════════════════════════════╝
  Measurement: RDTSC with LFENCE/RDTSCP serialisation
  Warmup: 3 reps · Measurement: 10 reps · Metric: min(cycles)
  Calibrated TSC frequency: 2.446 GHz

=== Memory Alignment Overhead Test ===
  Aligned ptr:   0x78a2051d3040 (aligned to 64B: YES)
  Misaligned ptr: 0x78a2051d3044 (aligned to 64B: NO)
  Aligned   GEMM 256x256         | min:    4547837 cy | time:    1.860 ms | GFLOPS:  18.04 | util:  46.1%
  Misaligned GEMM 256x256        | min:    4316704 cy | time:    1.765 ms | GFLOPS:  19.01 | util:  48.6%

=== GEMM Benchmarks (single-precision, C = A*B) ===
  Config                         | min cycles             | median cycles       | GFLOPS
  ------------------------------------------------------------------------------
  Scalar GEMM   64x  64          | min:     136979 cy | time:    0.056 ms | GFLOPS:   9.36 | util:  23.9%
  SIMD   GEMM   64x  64          | min:      32487 cy | time:    0.013 ms | GFLOPS:  39.47 | util: 100.9%
  Scalar GEMM  128x 128          | min:    5363417 cy | time:    2.193 ms | GFLOPS:   1.91 | util:   4.9%
  SIMD   GEMM  128x 128          | min:     478485 cy | time:    0.196 ms | GFLOPS:  21.44 | util:  54.8%
  Scalar GEMM  256x 256          | min:   50587502 cy | time:   20.685 ms | GFLOPS:   1.62 | util:   4.1%
  SIMD   GEMM  256x 256          | min:    4557563 cy | time:    1.864 ms | GFLOPS:  18.01 | util:  46.0%
  SIMD   GEMM  512x 512          | min:   52943471 cy | time:   21.648 ms | GFLOPS:  12.40 | util:  31.7%
  SIMD   GEMM 1024x1024          | min:  981704343 cy | time:  401.409 ms | GFLOPS:   5.35 | util:  13.7%

=== GeLU Benchmarks (element-wise, FP32) ===
  (excerpt)
  AVX2  GeLU n=   1024           | min:       2842 cy | time:    0.001 ms | Gelements/s: 13.22 | util:  33.8%
  AVX2  GeLU n=  16384           | min:      30919 cy | time:    0.013 ms | Gelements/s: 19.44 | util:  49.7%
  AVX2  GeLU n=1048576           | min:    2100899 cy | time:    0.859 ms | Gelements/s: 18.31 | util:  46.8%

=== Roofline Model Summary ===
  Peak FP32 throughput (1 core, FMA):      56 GFLOPS
  Peak memory bandwidth (assumed):         77 GB/s
  Arithmetic intensity (256x256 GEMM):     42.7 FLOP/byte
  Kernel regime:                           Compute-bound (AI > ridge point)

Performance matrix (summary): see `./bench` for full output.
```

Notes:
- These are single-core, single-threaded measurements. For system/multi-core comparisons, disable turbo and collect hardware counters.
- Store this file alongside the repository to keep a historical record of measured runs for reproducibility.
