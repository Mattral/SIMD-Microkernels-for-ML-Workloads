# Documentation

This repository now exposes structured documentation under `docs/` for architecture, setup, system design, and API details.

## Contents

* `architecture.md` — high-level architecture and module boundaries
* `system_design.md` — design rationale, data flow, and performance focus
* `setup.md` — build, test, and Python setup instructions
* `security.md` — security and supply-chain considerations for native and Python bindings
* `api/python_api.md` — Python API surface and type contract
* `design/gemm_algorithm.md` — blocked GEMM algorithm and kernel structure
* `design/avx2_register_file.md` — AVX2 register usage and micro-kernel layout
* `design/cache_hierarchy.md` — cache-aware tiling and memory locality strategy
* `design/performance_model.md` — roofline, arithmetic intensity, and benchmarking model
* `BENCHMARKS.md` — benchmark results and measurement notes

## How to use these docs

1. Review `setup.md` first to prepare the build environment.
2. Use `architecture.md` and `system_design.md` to understand the repo’s structure and design tradeoffs.
3. Consult `api/python_api.md` for Python bindings and `design/*` for low-level kernel details.
4. Run the benchmark script in `benchmarks/` and compare against the baseline in `docs/BENCHMARKS.md`.
