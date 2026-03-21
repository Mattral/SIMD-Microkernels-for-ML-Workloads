
# IntrinsicML: SIMD Microkernels for ML Workloads

Lightweight C++ implementations of SIMD-optimized microkernels for common ML primitives, with Python bindings and benchmarking utilities.

This project explores:

* Vectorized GEMM (FP32) using AVX2/FMA
* SIMD implementations of activation functions (GeLU)
* Cache-aware tiling and memory alignment
* Low-overhead benchmarking using CPU cycle counters

> ⚠️ This is an experimental project intended for learning and exploration. It is **not a replacement for production libraries** such as MKL, OpenBLAS, or oneDNN.

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
├── test_gemm.cpp
├── test_gelu.cpp
└── test_precision.py
```

---

## Build

### C++ Benchmark

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
./bench
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

