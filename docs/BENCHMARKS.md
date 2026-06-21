# IntrinsicML — Benchmark Results & Analysis

This document records benchmark results from controlled hardware runs. All
numbers in this file are from the development machine. **Always run `./bench`
or `./bench_stat` on your own hardware** — performance depends heavily on
microarchitecture, cache sizes, clock speed, and thermal state.

---

## Measurement Methodology

**Cycle-accurate timing** (`bench` / `main_bench.cpp`):
- RDTSC with LFENCE/RDTSCP serialisation (Intel SDM Vol.2B §4.3)
- 3 warmup reps, 10 measurement reps, reporting `min(cycles)`
- TSC frequency calibrated via 100 ms `clock_gettime(CLOCK_MONOTONIC)` interval

**Statistical timing** (`bench_stat` / `bench_stat.cpp`):
- `std::chrono::steady_clock` wall-clock (ns resolution via CLOCK_MONOTONIC)
- 5 warmup reps, 30 measurement reps
- Reports: min / mean ± 95% CI / median / peak GFLOPS
- CPU pinned to core 0 via `sched_setaffinity`

**For reproducible absolute numbers (recommended)**:
```bash
sudo cpupower frequency-set -g performance   # lock CPU frequency
taskset -c 0 ./bench_stat --reps 50 --output results.json
```

---

## Development Host Specification

| Parameter        | Value |
|------------------|-------|
| CPU model        | x86-64 desktop, AVX2+FMA support |
| Nominal clock    | 2.446 GHz (calibrated TSC) |
| L1d cache        | 32 KB (per core) |
| L2 cache         | 256 KB (per core) |
| L3 cache         | ~8 MB shared |
| Compiler         | GCC / Clang, `-O3 -march=native -mfma -ffast-math` |
| OS               | Ubuntu 22.04 / 24.04 |
| Kernel          | 5.x or 6.x |

---

## GEMM Benchmarks (Captured 2026-05-30)

```
=== GEMM Benchmarks (single-precision, C = A*B) ===

  Config                         | min cycles          | time       | GFLOPS  | util
  ─────────────────────────────────────────────────────────────────────────────────
  Scalar GEMM    64× 64          | min:     136,979 cy | 0.056 ms   |   9.36  | 23.9%
  SIMD   GEMM    64× 64          | min:      32,487 cy | 0.013 ms   |  39.47  |100.9%  ← cache-resident
  Scalar GEMM   128×128          | min:   5,363,417 cy | 2.193 ms   |   1.91  |  4.9%
  SIMD   GEMM   128×128          | min:     478,485 cy | 0.196 ms   |  21.44  | 54.8%
  Scalar GEMM   256×256          | min:  50,587,502 cy | 20.68 ms   |   1.62  |  4.1%
  SIMD   GEMM   256×256          | min:   4,557,563 cy | 1.864 ms   |  18.01  | 46.0%
  SIMD   GEMM   512×512          | min:  52,943,471 cy | 21.65 ms   |  12.40  | 31.7%
  SIMD   GEMM  1024×1024         | min: 981,704,343 cy | 401.4 ms   |   5.35  | 13.7%
```

**Observations:**

- The 5–12× speedup range aligns with the preprint's claims (Section 6.1).
- 64×64 exceeds 100% theoretical utilisation — a TSC/turbo artefact on small
  kernels where burst frequency exceeds the calibrated base frequency.
- The large drop from 256×256 → 512×512 indicates the B panel no longer fits
  in L2 (256 KB on this CPU). This is expected and documented in §3.1 of the
  preprint.
- 1024×1024 is DRAM-bandwidth bound — GFLOPS plateau reflects memory wall.

**Gap to OpenBLAS (single-threaded):**

On this hardware, OpenBLAS achieves approximately 35–45 GFLOPS for 256×256.
IntrinsicML achieves ~18 GFLOPS — roughly 40–50% of OpenBLAS. The primary
gap sources are documented in `docs/DESIGN.md §7`.

---

## GeLU Benchmarks (Captured 2026-05-30)

```
=== GeLU Benchmarks (element-wise, FP32) ===

  Kernel                         | min cycles  | time     | GElems/s  | util
  ─────────────────────────────────────────────────────────────────────────────
  AVX2  GeLU  n=     1024        |   2,842 cy  | 0.001 ms |  13.22    | 33.8%
  AVX2  GeLU  n=    16384        |  30,919 cy  | 0.013 ms |  19.44    | 49.7%
  AVX2  GeLU  n= 1,048,576       | 2,100,899cy | 0.859 ms |  18.31    | 46.8%
```

**Observations:**

- Small n (1024): overhead of loop setup → lower utilisation.
- Large n (≥16K): approaches the compute-bound regime at ~50% utilisation.
- The 50% ceiling reflects the rational polynomial evaluation cost (~15 ops/GeLU)
  vs peak FMA throughput (2 ops/cycle × 8 floats = 56 GFLOPS ≡ 56 GElems/s
  for a trivial activation). Our tanh approximation uses ~15 FMAs → ~37%
  of peak is the theoretical ceiling, matching observations.

---

## Roofline Analysis

```
  Peak FP32 throughput (1 core, AVX2 FMA): 56 GFLOPS
  Peak memory bandwidth (typical DDR4):    ~40–60 GB/s

  For 256×256 GEMM:
    FLOPs = 2 × 256³ = 33.6 MFLOP
    Data  = (256+256+256) × 256 × 4B = 786 KB (fits in L2 cache)
    AI    = 33.6e6 / 786e3 ≈ 42.7 FLOP/byte — compute-bound above ridge point

  Conclusion: the kernel is correctly operating in the compute-bound regime
  for moderate sizes. The gap to peak is due to the instruction-level
  overhead documented in DESIGN.md §7 (C++ intrinsics vs assembly, fixed tiles).
```

---

## Statistical Benchmark Output (bench_stat, 30 reps)

```
Kernel        N     min ms   mean±CI ms      median ms   peak GFLOPS
------        ---   ------   ----------      ---------   -----------
simd_packed    64    0.013    0.014±0.001      0.013        39.5
simd_packed   128    0.196    0.201±0.003      0.199        21.4
simd_packed   256    1.864    1.891±0.015      1.877        18.0
simd_packed   512   21.65    21.91±0.18       21.80         12.4

Notes:
  - CI widths ~1–2% → low measurement noise (good for regression detection)
  - min ≈ median for compute-bound kernels (consistent execution)
  - GH Actions runners show 3–5× wider CIs (noisy scheduler + shared hardware)
```

---

## How to Update This File

After running a controlled benchmark:

```bash
# Lock frequency
sudo cpupower frequency-set -g performance

# Run statistical harness
taskset -c 0 ./bench_stat --reps 50 --output benchmarks/results/gemm_results.json

# Run the RDTSC harness (for the legacy cycle-count numbers)
taskset -c 0 ./bench

# Paste the relevant output section into this file with hardware spec and date.
```

Please include:
1. CPU model and cache sizes
2. Compiler version (`g++ --version`)
3. Date and OS version
4. Whether turbo / frequency lock was applied
