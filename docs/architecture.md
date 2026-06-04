# Architecture

This repository is organized around three main subsystems:

* `src/kernels/` — SIMD microkernels for matrix multiplication and activation functions
* `src/bindings/` — Python-facing wrappers using `pybind11`
* `benchmarks/` and `src/main_bench.cpp` — measurement harness and automation

## Core components

### GEMM kernel

The primary compute kernel is a blocked FP32 matrix multiply implemented in `src/kernels/avx_matmul.cpp`.
It uses AVX2 intrinsics and fused multiply-add (FMA) to maximize throughput for inner loops.

### Activation kernels

`src/kernels/intrinsic_gelu.cpp` provides a vectorized GeLU implementation using the standard tanh-based approximation.
The code is branch-free and uses AVX2 registers to process 8 floats per vector.

### Memory allocation

`src/kernels/cache_alloc.hpp` provides aligned allocation helpers to ensure 64-byte alignment for SIMD-friendly loads and stores.
This reduces cache-line split penalties and supports efficient vectorized accesses.

## Python bindings

`src/bindings/pybind_entry.cpp` exposes the core compute functions to Python.
The API is designed for convenience:

* `sgemm(A, B, C=None)` — performs FP32 GEMM with optional output buffer allocation
* `gelu(X)` — applies the GeLU activation to a NumPy-compatible array
* `build_info()` — returns build metadata as a Python dict

The binding layer validates shapes, preserves alignment, and minimizes copies when possible.

## Benchmark harness

`src/main_bench.cpp` is the project’s benchmark entry point.
It measures cycles and time with serialized `RDTSC`/`RDTSCP` timestamps, performs repeated runs, and can emit JSON reports for structured analysis.

Optional OpenMP support is enabled through the CMake option `-DSIMD_ML_OPENMP=ON`.
When enabled, the GEMM benchmark uses thread-level parallelism to exercise the multi-core path.

## Build system

The project uses CMake with separate targets for:

* `bench` — benchmark executable
* `simd_tests` / `simd_tests_doctest` — native C++ tests
* Python extension build via `pip install -e .`

The repo is structured to keep low-level kernels independent from the benchmark and binding scaffolding.
