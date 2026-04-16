
# IntrinsicML: SIMD Microkernels for ML Workloads

Lightweight C++ implementations of SIMD-optimized microkernels for ML primitives, with Python bindings, benchmark automation, and optional OpenMP support.

This project explores:

* Vectorized GEMM (FP32) using AVX2/FMA
* SIMD implementations of activation functions (GeLU, ReLU, SiLU, Softmax)
* Cache-aware blocking and memory alignment
* Low-overhead benchmarking with JSON output
* Optional OpenMP multithreading for GEMM

This repository is intended as an educational and experimental reference for low-level kernel design. It is not intended as a drop-in replacement for optimized production BLAS libraries.

---

## Overview

Modern ML workloads rely heavily on dense linear algebra and elementwise operations. While highly optimized libraries exist, implementing simplified kernels from first principles is a useful way to understand:

* SIMD execution (AVX2 / AVX-512)
* Cache hierarchy and data locality
* Instruction-level parallelism (ILP)
* Trade-offs between compute and memory bandwidth

This repository provides small, self-contained implementations of these ideas.

---

## Project Structure

```
src/
├── kernels/
│   ├── avx_matmul.cpp       # Tiled FP32 GEMM (AVX2/FMA)
│   ├── intrinsic_gelu.cpp   # Vectorized GeLU approximation
│   └── cache_alloc.hpp      # Aligned allocation helpers
├── bindings/
│   └── pybind_entry.cpp     # Python bindings (pybind11)
└── main_bench.cpp           # Benchmark harness (RDTSC-based)

tests/
├── test_alignment.cpp
├── test_gelu.cpp
├── test_gemm.cpp
├── test_gemm_packed.cpp
├── test_threads.cpp
└── test_doctest.cpp

benchmarks/
├── run_bench.sh
└── results/
    └── bench_results.json

DESIGN.md
BENCHMARKS.md
CITATION.cff
```

---

## Build

### C++ Benchmark

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMD_ML_OPENMP=ON
cmake --build . --parallel
./bench --json ../benchmarks/results/bench_results.json
```

Or run the packaged benchmark automation script:

```bash
./benchmarks/run_bench.sh
```

### Python Extension

```bash
pip install -e .
python -c "import simd_kernels; print(simd_kernels.build_info())"
```

---

## Implementation Highlights

### Memory Alignment

All arrays are allocated with 64-byte alignment to match cache line size:

* Reduces split-load penalties
* Improves SIMD load/store efficiency

Implemented via `posix_memalign` in `cache_alloc.hpp`.

---

### Tiled GEMM (AVX2)

The matrix multiplication kernel uses:

* **Blocking/tiling** to improve cache locality
* **SIMD vectorization** via `_mm256_*` intrinsics
* **FMA instructions** for combined multiply-add

The implementation is intentionally simple:

* No packing stage
* Limited prefetching
* Fixed tile sizes

This keeps the code readable while still demonstrating performance gains over naïve loops.

---

### SIMD GeLU Approximation

The GeLU activation is implemented using a tanh-based approximation:

```
GeLU(x) ≈ 0.5 x (1 + tanh(√(2/π)(x + 0.044715x³)))
```

Vectorized using AVX2 intrinsics:

* Polynomial evaluation in SIMD registers
* Branch-free implementation
* Approximate reciprocal instead of division

Accuracy is validated against a scalar reference.

---

### Benchmarking Approach

The benchmark harness uses `RDTSC` to estimate cycle counts.

Key details:

* Measurements use minimum of multiple runs
* Designed to reduce noise from OS scheduling

> ⚠️ Results should be interpreted as **approximate**:
>
> * No CPU pinning
> * No frequency locking
> * No statistical analysis beyond min-of-N

The current `bench` harness in this repo implements the P0 remediation
prescriptions: a short TSC calibration to estimate ticks/second, LFENCE/RDTSCP
serialised timestamps, and optional CPU affinity pinning. The harness reports
per-configuration cycle counts, elapsed milliseconds, achieved GFLOPS, and a
utilisation percentage relative to a per-core theoretical peak.

For rigorous benchmarking, external tools (e.g., `perf`, VTune) are recommended.

---

## Performance (Illustrative)

Example results on a desktop x86 CPU (AVX2 enabled):

| Size | Naïve C++   | SIMD Kernel | Speedup |
| ---- | ----------- | ----------- | ------- |
| 128³ | ~8× slower  | baseline    | ~5–8×   |
| 256³ | ~10× slower | baseline    | ~8–10×  |

Observations:

* SIMD + tiling significantly outperform naïve triple-loop
* Performance remains below optimized BLAS libraries
* Larger matrices benefit more from cache reuse

> These numbers are indicative only and depend heavily on hardware and compiler.

---

## Limitations

This project intentionally omits many techniques used in production libraries:

* No matrix packing (critical for high-performance GEMM)
* No architecture-specific tuning (e.g., Skylake vs Zen)
* No multithreading
* No NUMA awareness
* No comparison against MKL/OpenBLAS included yet

As a result, performance is **educational, not state-of-the-art**.

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

## When to Use This

Useful for:

* Learning SIMD programming with intrinsics
* Understanding GEMM structure
* Experimenting with low-level optimizations

Not suitable for:

* Production ML workloads
* Competitive benchmarking vs optimized libraries

---

## Future Work

* Add packed GEMM path
* Compare against OpenBLAS / Eigen
* Add AVX-512 specialization
* Improve benchmarking methodology (pinning, perf counters)
* Explore multithreaded execution

---

## License

MIT

---

