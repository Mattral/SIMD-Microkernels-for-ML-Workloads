# System Design

This project is designed to demonstrate a working SIMD microkernel pipeline while keeping the codebase accessible for review.

## Design goals

* Deliver a correct, vectorized FP32 GEMM kernel with AVX2/FMA
* Provide a minimal Python integration path for experimentation
* Expose benchmark instrumentation with reproducible JSON output
* Keep the core kernel independent of framework-specific dependencies

## Data flow

1. Input matrices are allocated with 64-byte aligned storage.
2. The GEMM driver loops over `M`, `N`, and `K` blocks.
3. Each tile is processed by an AVX2-based inner kernel.
4. The output is written back to the result matrix in row-major order.

The activation path follows a similar flow, with vectorized processing of contiguous elements through `gelu`.

## Performance strategy

The implementation uses fixed-size tiling and blocked loops to improve cache reuse.
The current design intentionally avoids a full packed-GEMM pipeline so that the inner kernel and blocking strategy remain visible and easy to review.

### Threading

OpenMP is used only as an optional build-time feature.
When enabled, the benchmark harness exercises a multithreaded path for GEMM, but the core compute kernel remains thread-agnostic.

### Runtime dispatch

This repository targets AVX2 by default.
The code is compiled for x86 with AVX2 support and does not perform dynamic ISA dispatch at runtime.

## Build-time configuration

* `SIMD_ML_OPENMP=ON` — enable OpenMP threading
* `CMAKE_BUILD_TYPE=Release` — enable compiler optimizations
* `-march=native` or explicit `-mavx2 -mfma` may be used to tune for the host CPU

## Documentation and review

The `docs/` tree is intended to support system-level review by explaining architecture, build setup, API behavior, and security considerations in a single place.
