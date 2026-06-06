# IntrinsicML ⚡

> Hand-rolled SIMD microkernels for ML primitives — AVX2 GEMM, vectorized GeLU, Python bindings, and cycle-accurate benchmarks. Built from first principles to understand what happens *below* the BLAS layer.

[![CI](https://github.com/Mattral/SIMD-Microkernels-for-ML-Workloads/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/SIMD-Microkernels-for-ML-Workloads/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-informational?logo=cplusplus)](CMakeLists.txt)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-informational?logo=python)](pyproject.toml)

---

## Why this exists

Production ML runs on MKL, cuBLAS, and oneDNN. Those libraries are heavily optimized black boxes. This project exists to answer: **what are the minimum ingredients needed to get 5–10× speedup over naïve C++ using only CPU intrinsics?**

This is an educational implementation — intentionally simple, thoroughly commented, and honest about its limitations. It's a useful reference if you're learning:

- SIMD programming with AVX2/FMA intrinsics
- Cache-aware tiling and memory alignment
- How activation functions like GeLU can be vectorized
- Low-overhead cycle-count benchmarking with `RDTSC`

---

## What's implemented

| Kernel | Instruction Set | Notes |
|---|---|---|
| Tiled FP32 GEMM | AVX2 + FMA | Blocking for L2 cache; no packing stage |
| GeLU activation | AVX2 | tanh-based polynomial approximation |
| Aligned allocator | — | 64-byte alignment via `posix_memalign` |
| Python bindings | — | pybind11; importable as `simd_kernels` |

---

## Performance

Measured on a desktop x86 CPU with AVX2 support. Min-of-N runs via RDTSC.

| Matrix size | Naïve triple-loop | SIMD + tiling | Speedup |
|---|---|---|---|
| 128 × 128 × 128 | ~40 ms | ~5–8 ms | **5–8×** |
| 256 × 256 × 256 | ~320 ms | ~32–40 ms | **8–10×** |

**Context:** These numbers are real but not rigorous — no CPU pinning, no frequency locking. For comparison, OpenBLAS achieves roughly 2–3× more than this project's SIMD kernel at the same sizes, primarily because of matrix packing and prefetching that this project omits by design.

---

## Getting started

### Build the C++ benchmark

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
./bench
```

Requires: CMake ≥ 3.16, a compiler with AVX2 support (GCC 9+ or Clang 10+).

### Install the Python extension

```bash
pip install -e .
python -c "import simd_kernels; print(simd_kernels.build_info())"
```

### Run tests

```bash
# C++ unit tests
cd build && ctest --output-on-failure

# Python precision check
python tests/test_precision.py
```

---

## Project structure

```
src/
├── kernels/
│   ├── avx_matmul.cpp       # Tiled FP32 GEMM (AVX2/FMA)
│   ├── intrinsic_gelu.cpp   # Vectorized GeLU approximation
│   └── cache_alloc.hpp      # Aligned allocation helpers
├── bindings/
│   └── pybind_entry.cpp     # Python bindings (pybind11)
└── main_bench.cpp           # Benchmark harness (RDTSC)

tests/
├── test_gemm.cpp
├── test_gelu.cpp
└── test_precision.py
```

---

## Implementation notes

### Tiled GEMM
Matrix multiplication is tiled so the working set of each block fits in L2 cache. Each inner loop uses `_mm256_fmadd_ps` (fused multiply-add) to process 8 floats per cycle. No packing is done — which is the main reason this doesn't match BLAS performance.

### SIMD GeLU
Implements the tanh approximation:

```
GeLU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
```

Polynomial evaluation and tanh approximation are done entirely in AVX2 registers — no branches, no scalar fallback.

### Benchmarking
The harness uses `RDTSC` (Read Time-Stamp Counter) for sub-microsecond timing. The reported result is the minimum across N iterations to reduce OS scheduling noise. This is appropriate for short microbenchmarks but not a substitute for `perf` or VTune for production analysis.

---

## Honest limitations

This project deliberately skips techniques that production libraries rely on:

- **No matrix packing** — the biggest missing piece for GEMM performance
- **No prefetching** — data is not explicitly brought into cache ahead of compute
- **No multithreading / OpenMP** — kernels are single-threaded
- **No AVX-512** — only AVX2 is targeted
- **No architecture tuning** — tile sizes are fixed, not auto-tuned per CPU

These aren't bugs; they're scope decisions that keep the code readable.

---

---


Observations:

* SIMD + tiling significantly outperform naïve triple-loop
* Performance remains below optimized BLAS libraries
* Larger matrices benefit more from cache reuse

> These numbers are indicative only and depend heavily on hardware and compiler.

---

## Citation

This repository includes a `CITATION.cff` manifest so the project can be cited consistently in academic or engineering reports.

## Cache-Arithmetic Derivation (example)

To reason about whether a GEMM kernel is memory- or compute-bound we use a
simple arithmetic-intensity model. For an $M\times K\times N$ GEMM the total
work is $2\cdot M\cdot N\cdot K$ FLOPs. Minimal traffic to read inputs and
write outputs (FP32) is approximately $4\cdot(MK + KN + MN)$ bytes. The
arithmetic intensity (AI) is therefore

$$
\mathrm{AI} = \frac{2 M N K}{4 (M K + K N + M N)}\ \mathrm{FLOP/byte}.
$$

For 256×256×256 this evaluates to ~$42.7$ FLOP/byte which is well into the
compute-bound regime on most desktop/server CPUs — hence improving the inner
kernel and packing typically yields better GFLOPS than micro-optimising loads.

## Roofline & Benchmarking Methodology

This repository includes a compact roofline-style summary in the `bench`
output. The summary compares measured per-core GFLOPS against a conservative
single-core peak (AVX2 FMA: 16 FP32 ops per cycle per core × measured Hz).
`BENCHMARKS.md` contains verbatim runs captured on the host used during
development. For reproducible tables, run:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
./bench
```

Interpret `bench` results as per-core, single-threaded throughput. For system
wide peak comparisons or multithreaded experiments, use a controlled
environment (disable turbo, pin threads, collect hardware counters with `perf`).

---

---

## Roadmap

- [ ] Add packed GEMM path and measure the delta
- [ ] Direct comparison vs. OpenBLAS and Eigen at equal matrix sizes
- [ ] AVX-512 specialization (F16C, VNNI)
- [ ] Multithreaded GEMM with OpenMP
- [ ] Rigorous benchmarks: CPU pinning, frequency locking, confidence intervals

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
