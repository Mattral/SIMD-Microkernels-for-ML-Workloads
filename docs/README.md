# Documentation Index

---

## Getting Started

- **[setup.md](setup.md)** — Build prerequisites, CMake configuration, Python extension install, Docker, and reproducible benchmarking
- **[api/python_api.md](api/python_api.md)** — Complete Python API reference for all 10 exported functions with type signatures, examples, and error messages

---

## Architecture and Design

- **[architecture.md](architecture.md)** — Source layout, component diagram, and responsibilities of each subsystem
- **[system_design.md](system_design.md)** — Data-flow diagrams (GEMM and GeLU paths), memory allocation strategy, ISA dispatch, threading model
- **[DESIGN.md](DESIGN.md)** — Deep-dive into every major design decision: cache-blocking derivation, GeLU polynomial choice, LayerNorm pass structure, Python binding patterns, CI architecture
- **[ROADMAP.md](ROADMAP.md)** — Versioned milestone history and planned work; honest limitations table

---

## Kernel Design Reference

- **[design/gemm_algorithm.md](design/gemm_algorithm.md)** — Goto/BLIS 5-loop structure, panel packing layouts, 8×8 and 6×16 micro-kernel comparison
- **[design/avx2_register_file.md](design/avx2_register_file.md)** — YMM register allocation, FMA throughput analysis, AVX-512 status
- **[design/cache_hierarchy.md](design/cache_hierarchy.md)** — Cache-level targets per loop, tile-size derivation, software prefetch distance rationale
- **[design/performance_model.md](design/performance_model.md)** — Theoretical peak GFLOPS, arithmetic intensity / roofline, benchmark methodology (`bench` vs `bench_stat`), interpreting utilisation output

---

## Benchmark Results

- **[BENCHMARKS.md](BENCHMARKS.md)** — Captured performance numbers, roofline analysis, statistical benchmark output, instructions for updating the baseline
