# IntrinsicML — SIMD Microkernels for ML Workloads
# GitHub Copilot Upgrade Instructions — FAANG / Frontier-Lab Standard
# Target: v2.0 | Systems Engineer / ML Compiler Hiring Bar

---

## STRATEGIC BRIEF — READ BEFORE ANY PROMPT BLOCK

### Honest state of the repo today

The repo is C++ 68.8% / Python 27.1% / CMake 4.1%.
It has 6 commits and 6 stars. Zero forks.

What exists is genuinely good scaffolding:
- CMakeLists.txt with AVX2/AVX-512 detection, correct `-march=native -O3 -mfma` flags
- RDTSC-based benchmarking harness
- pybind11 Python bindings
- Tests directory (structure exists, need to verify depth)

What the README honestly admits is missing and is killing this repo's signal:
- No matrix packing (Goto/BLIS algorithm — the #1 missing piece)
- No multithreading (OpenMP is 4 lines to add)
- No comparison against OpenBLAS/Eigen (the most important table for any
  systems engineer reviewing this)
- Benchmarking has no CPU pinning, no frequency locking, no statistical rigor
- README explicitly says "educational, not state-of-the-art" — this is
  self-defeating for a portfolio repo

The goal of this upgrade: **remove every self-deprecating disclaimer by
actually fixing the underlying gaps.** A FAANG systems reviewer should see
a repo that competes seriously with OpenBLAS on single-core throughput via
a documented, principled approach.

---

## ANSWERED DESIGN QUESTIONS

### 100% test coverage?

**No — target 95% line coverage on kernel source, 100% on public API surface.**
For systems code the meaningful tests are:
1. Numerical correctness (bit-for-bit agreement with reference for small sizes)
2. Performance regression gates (throughput must not drop >5% vs baseline)
3. Edge cases (non-multiple-of-8 dimensions, minimum sizes, alignment cases)
4. UB detection (run tests under AddressSanitizer and UBSanitizer in CI)

### Is CI relevant?

**Absolutely yes — more than any other repo in your portfolio.** Reasons:
- C++ code has undefined behaviour and alignment bugs that are hardware-
  specific. CI on multiple compiler versions catches this.
- Performance regression tests need to run on a consistent machine — use
  a self-hosted runner or GitHub-hosted large runner for benchmarks.
- Cross-platform: AVX2 is x86 only. CI must detect and skip SIMD paths on
  ARM (e.g., Apple Silicon GitHub runners) gracefully.

Two CI workflows: `build-and-test.yml` (every PR) + `bench.yml` (weekly
scheduled + main pushes, uploads results as artefacts).


The right infra additions are: **Docker** (reproducibility), **GitHub Actions**
(CI), and **perf / VTune** integration in the benchmark harness (credibility).


**Note:** add a thin Rust `safe_wrapper` crate using PyO3 that wraps the
C++ library via FFI as a second-surface (1–2 days of work). This shows you
understand cross-language FFI without abandoning C++. The Rust wrapper is
the ONLY Rust code in this repo.

### How to evaluate and benchmark?

See BLOCK 5 (benchmark harness) in full detail. Short answer:
- Primary metric: **GFLOPS** (not "speedup" — GFLOPS is hardware-independent
  and comparable across machines)
- Baseline: OpenBLAS SGEMM (single-threaded, then multi-threaded)
- Secondary: Eigen, naive triple-loop, naïve auto-vectorised
- Methodology: CPU frequency lock + core pinning + 100 warm-up iterations
  + median of 1000 trials + report p50/p95/p99
- Report % of theoretical peak (= 2 × frequency × AVX2 FMA lanes × cores)
  This is how BLIS/OpenBLAS papers report performance and it is what
  hardware engineers understand immediately

### How to be top-tier?

Four moves, in priority order:
1. **Implement packed GEMM** (Goto algorithm). This is the single biggest
   technical gap and the most important algorithmic contribution. Without it
   the current kernel is a demo, not a kernel.
2. **Add OpenBLAS comparison table** with actual measured GFLOPS committed
   to the repo. This changes the entire narrative.
3. **Add AVX-512 path** and a runtime ISA dispatcher that selects AVX-512 vs
   AVX2 vs SSE4.2 at startup. This is what production kernels do.
4. **Add multithreading via OpenMP** with a performance model showing how
   parallelism interacts with cache hierarchy.

---

## TARGET REPOSITORY STRUCTURE

```
IntrinsicML/
│
├── src/
│   ├── kernels/
│   │   ├── gemm/
│   │   │   ├── naive_gemm.hpp          # Scalar reference (correctness oracle)
│   │   │   ├── avx2_gemm.cpp           # AVX2 GEMM (no packing — current)
│   │   │   ├── avx2_gemm_packed.cpp    # AVX2 GEMM with packing (NEW — P0)
│   │   │   ├── avx512_gemm_packed.cpp  # AVX-512 path (NEW)
│   │   │   └── gemm_dispatcher.cpp     # Runtime ISA dispatch (NEW)
│   │   ├── activations/
│   │   │   ├── gelu_avx2.cpp           # Vectorised GeLU (current)
│   │   │   ├── gelu_avx512.cpp         # AVX-512 GeLU (NEW)
│   │   │   ├── relu_avx2.cpp           # NEW
│   │   │   ├── silu_avx2.cpp           # NEW (used in LLaMA etc.)
│   │   │   └── softmax_avx2.cpp        # NEW (row-wise online softmax)
│   │   ├── attention/
│   │   │   └── flash_attention_v1.cpp  # Single-head FlashAttention CPU (NEW)
│   │   ├── quantization/
│   │   │   └── int8_quant.cpp          # Symmetric int8 quantization (NEW)
│   │   └── cache_alloc.hpp             # 64-byte aligned allocator (current)
│   ├── bindings/
│   │   └── pybind_entry.cpp            # Python/pybind11 bindings (expand)
│   ├── dispatch/
│   │   ├── cpuid.hpp                   # CPUID feature detection (NEW)
│   │   └── kernel_registry.hpp        # Dispatch table (NEW)
│   └── main_bench.cpp                  # CLI benchmark harness (upgrade)
│
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/
│   │   ├── test_gemm_correctness.cpp
│   │   ├── test_gemm_packed_correctness.cpp
│   │   ├── test_gelu_correctness.cpp
│   │   ├── test_activations_correctness.cpp
│   │   ├── test_int8_quant.cpp
│   │   └── test_dispatcher.cpp
│   ├── precision/
│   │   └── test_ulp_error.cpp          # Max ULP error vs reference
│   ├── performance/
│   │   └── test_perf_regression.py     # Python perf gate (via simd_kernels)
│   └── python/
│       └── test_precision.py           # Existing — upgrade
│
├── benchmarks/
│   ├── bench_gemm.cpp                  # GFLOPS across sizes vs OpenBLAS
│   ├── bench_activations.cpp           # GB/s for activation functions
│   ├── bench_e2e_mlp.cpp               # End-to-end MLP forward pass
│   ├── run_suite.sh                    # Pin CPU, disable turbo, run all
│   └── results/
│       ├── gemm_results.json           # Committed baseline (your machine)
│       └── README.md                   # How to regenerate
│
├── docs/
│   ├── design/
│   │   ├── gemm_algorithm.md           # Goto GEMM with diagrams
│   │   ├── avx2_register_file.md       # Register blocking explanation
│   │   ├── cache_hierarchy.md          # Cache model and tiling rationale
│   │   └── performance_model.md        # Roofline model for this hardware
│   └── api/
│       └── python_api.md
│
├── rust_wrapper/                       # Optional thin Rust/PyO3 surface
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
│
├── .github/
│   └── workflows/
│       ├── build-and-test.yml          # Every PR
│       └── bench.yml                   # Weekly + main push
│
├── Dockerfile
├── CMakeLists.txt                      # Upgrade existing
├── pyproject.toml                      # Already exists
├── BENCHMARKS.md                       # Committed results with methodology
├── DESIGN.md                           # Algorithm design decisions
├── CONTRIBUTING.md
├── CITATION.cff
└── README.md                           # Full overhaul
```

---

## CONTEXT BRIEF (paste at top of every Copilot session)

```
Repository: https://github.com/Mattral/SIMD-Microkernels-for-ML-Workloads
Language: C++17, Python 3.10+
Build: CMake 3.22+, pybind11, OpenBLAS (for benchmarking comparison only)
CI: GitHub Actions (.github/workflows/)
Tests: CTest (C++ unit tests) + pytest (Python precision tests)
Primary metric: GFLOPS (FP32 single-threaded) vs OpenBLAS baseline
Target ISA: AVX2/FMA (primary), AVX-512 (secondary), SSE4.2 (fallback)
Known gaps being fixed in this session: [PASTE RELEVANT BLOCK TITLE]
```

---

## BLOCK 1 — Packed GEMM: the Goto algorithm (P0 — most important)

**Create file:** `src/kernels/gemm/avx2_gemm_packed.cpp`
**Create file:** `src/kernels/gemm/avx2_gemm_packed.hpp`
completed

**Acceptance criteria:**
- `ctest -R test_gemm_packed_correctness` passes for M,N,K in
  {1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024}
- Max absolute error vs naive_gemm reference < 1e-4 for all sizes
- GFLOPS on 512×512 is ≥ 60% of naive GEMM on the same machine
  (demonstrates packing benefit — not targeting OpenBLAS yet)

---

## BLOCK 2 — Runtime ISA dispatcher

**Create files:** `src/dispatch/cpuid.hpp`, `src/dispatch/kernel_registry.hpp`,
  `src/kernels/gemm/gemm_dispatcher.cpp`

**Status:** runtime dispatcher implemented and Python binding integration verified.

completed

**Acceptance criteria:**
- On an AVX2 machine, `python -c "import simd_kernels; print(simd_kernels.detected_isa())"` prints "avx2" or "avx512".
- `ctest -R test_dispatcher` passes.
- On an ARM machine (GitHub Actions ubuntu-arm runner), the library compiles and the scalar fallback is selected.

---

## BLOCK 3 — New activation kernels: ReLU, SiLU, row-wise Softmax

**Create files:** `src/kernels/activations/relu_avx2.cpp`,
  `src/kernels/activations/silu_avx2.cpp`,
  `src/kernels/activations/softmax_avx2.cpp`

**Prompt:**
Done

---

## BLOCK 4 — Multithreading with OpenMP

**Files to modify:** `src/kernels/gemm/avx2_gemm_packed.cpp`, `CMakeLists.txt`

**Prompt:**

```
Add OpenMP parallelism to the packed GEMM kernel. This is the critical
missing feature that transforms the kernel from single-core to multi-core.



---

## BLOCK 5 — Production benchmark harness with statistical rigor

**Create files:** `benchmarks/bench_gemm.cpp`, `benchmarks/run_suite.sh`,
  `benchmarks/results/gemm_results.json`



---

## BLOCK 6 — C++ unit tests: correctness + AddressSanitizer + UBSan

**Create files:** `tests/CMakeLists.txt` (upgrade),
  `tests/unit/test_gemm_correctness.cpp`

**Prompt:**

```
Create a comprehensive C++ unit test suite using Google Test (or doctest —
lighter, header-only, preferred for microkernel repos).

Use doctest (https://github.com/doctest/doctest) — add as a FetchContent
dependency in tests/CMakeLists.txt.



---

## BLOCK 7 — CI workflows: build-and-test + bench

**Create files:** `.github/workflows/build-and-test.yml`,
  `.github/workflows/bench.yml`

Completed

---

## BLOCK 8 — Dockerfile for reproducible builds

Completed

---

## BLOCK 9 — Documentation: DESIGN.md and algorithm comments

**Create file:** `DESIGN.md`

**Prompt:**

```
Create a DESIGN.md that explains the mathematical and architectural decisions
at a level appropriate for a systems engineer or compiler team interview.

DESIGN.md sections:

## 1. The GEMM Performance Problem
- Why naive triple-loop is slow (GFLOPS << peak)
- The roofline model: compute-bound vs memory-bandwidth-bound
- For FP32 GEMM at large N: arithmetic intensity = N/2 FLOP/byte →
  becomes compute-bound for N > 128 (show the calculation)

## 2. The Goto Algorithm
- Three-level cache tiling rationale (with the actual formulas for MC/NC/KC)
- Why packing eliminates TLB thrashing
- Diagram: data flow through L1/L2/L3 cache during GEMM
- The 8×8 register block: why 8 is the right number for AVX2
  (16 YMM registers − 2 for A/B loads = 14 accumulators → round to 8×1 or 4×2)

## 3. AVX2 Register Blocking
- YMM register layout for FP32 GEMM
- FMA latency hiding: why the accumulator dimension is > 1
  (FMA latency = 4 cycles on Haswell/Zen → need 4 independent accumulators
   to saturate the FMA port at 1 FMA/cycle throughput)
- Show the inner kernel structure: 8 accumulators, 2 loads per iter

## 4. Activation Function Numerics
- GeLU approximation error budget (ULP analysis)
- SiLU exp approximation: range reduction + Horner polynomial
- Softmax numerical stability: why subtract max (avoid overflow)

## 5. Performance Model
- Theoretical peak: 2 × freq × lanes × fma_depth
  For 3.6 GHz / AVX2 / single-core: 3.6 × 16 = 57.6 GFLOPS
- Expected efficiency: ~65–75% of peak with good packing
- Comparison with OpenBLAS: OpenBLAS uses prefetching and architecture-tuned
  tile sizes; we aim for 70–80% of OpenBLAS as an educational target

## 6. What Is Missing vs Production
- Hardware prefetching hints (PREFETCHNTA/PREFETCHT0)
- NUMA-aware allocation
- Architecture-specific tuning (Skylake vs Zen4 have different cache sizes)
- Int8/BF16 kernels (increasingly important for LLM inference)
- AMX (Advanced Matrix Extensions) for future work

Also update every source file header comment to reference the relevant
DESIGN.md section number. Example:

avx2_gemm_packed.cpp header:
  // Packed SGEMM using the Goto/BLIS algorithm.
  // See DESIGN.md §2 for algorithm rationale and §3 for register blocking.
  // Tile sizes (MC=128, KC=256, NC=2048) chosen per DESIGN.md §2 cache model.
```


---

## BLOCK 11 — README overhaul and BENCHMARKS.md

**Files to replace:** `README.md`
**Create:** `BENCHMARKS.md`, `CITATION.cff`

**Prompt:**

```
Rewrite README.md for FAANG/systems-engineer audience. Remove ALL
self-deprecating disclaimers. Replace them with honest, specific claims
backed by committed benchmark results.

README.md structure:

## Badges (row 1)
  CI | License | C++ Standard | AVX2/AVX-512 | Python | Stars

## Tagline (1 sentence)
  "A from-scratch implementation of packed SGEMM, GeLU, SiLU, Softmax,
   and online attention using AVX2/AVX-512 intrinsics — achieving ~70%
   of OpenBLAS single-core throughput via the Goto algorithm."

## Performance (lead with this — it is what engineers look at first)
  Show the benchmark table from BENCHMARKS.md.
  Use actual committed numbers, NOT "illustrative".
  Column: Size | Our GFLOPS | OpenBLAS GFLOPS | % of Peak | % of OpenBLAS
  Note: "Benchmarked on [CPU]. Regenerate: bash benchmarks/run_suite.sh"

## What This Implements
  Table: Kernel | ISA | Status | Paper/Reference
  avx2_gemm_packed | AVX2/FMA | ✅ | Goto & van de Geijn 2008
  avx2_gemm_packed | AVX-512 | ✅ | ...
  gelu_avx2 | AVX2 | ✅ | Hendrycks & Gimpel 2016
  silu_avx2 | AVX2 | ✅ | Ramachandran et al. 2017
  softmax_avx2 | AVX2 | ✅ | Milakov & Gimelshein 2018
  flash_attention_v1 | AVX2 | 🚧 | Dao et al. 2022

## Algorithm Design
  "See DESIGN.md for the roofline model, cache tiling rationale,
   and register blocking analysis."
  (Two paragraph summary, then link)

## Quick Start
  git clone ...
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . --parallel
  ./bench_gemm

  # Python:
  pip install -e .
  python -c "import simd_kernels; print(simd_kernels.detected_isa())"

## Limitations (honest, specific, not dismissive)
  "Compared to OpenBLAS and MKL, this implementation omits: hardware
   prefetch hints, architecture-specific tile tuning (we use static sizes
   rather than runtime L-cache probing), and NUMA-aware allocation.
   These gaps account for the remaining 20–30% gap vs OpenBLAS.
   See DESIGN.md §6 for details."

## References
  [1] Goto & van de Geijn (2008) — The Goto algorithm (DESIGN.md §2)
  [2] BLIS framework: Smith et al. (2014)
  [3] Hendrycks & Gimpel (2016) — GeLU
  [4] Dao et al. (2022) — FlashAttention

FILE: CITATION.cff
  cff-version: "1.2.0"
  title: "IntrinsicML: SIMD Microkernels for ML Workloads"
  authors: [{family-names: "Myet", given-names: "Min Htet"}]
  version: "2.0.0"
  license: MIT
  repository-code: "https://github.com/Mattral/SIMD-Microkernels-for-ML-Workloads"
  keywords: [simd, avx2, avx-512, gemm, machine-learning, microkernels, cpp]
  preferred-citation:
    type: software
    title: IntrinsicML
    doi: (add Zenodo DOI after publishing)
```

---

## MASTER CHECKLIST

Create as GitHub Issue "v2.0.0 upgrade tracker":

**P0 — Core algorithmic gaps (do first, they change the repo's identity)**
- [x] BLOCK 1: Packed GEMM (Goto algorithm) — avx2_gemm_packed.cpp
- [x] BLOCK 2: Runtime ISA dispatcher (cpuid + kernel_registry)
- [x] BLOCK 5: Production benchmark harness (GFLOPS + OpenBLAS comparison)
- [x] Commit actual benchmark results to benchmarks/results/

**P1 — Completeness**
- [x] BLOCK 3: ReLU, SiLU, Softmax kernels
- [x] BLOCK 4: OpenMP multithreading
- [x] BLOCK 6: C++ unit tests with doctest + ASan/UBSan
- [x] BLOCK 7: CI workflows (build-and-test + bench)

**P2 — Polish**
- [x] BLOCK 8: Dockerfile
- [x] BLOCK 9: DESIGN.md
- [x] BLOCK 10: Expanded pybind11 bindings + .pyi type stubs
- [x] BLOCK 11: README overhaul + BENCHMARKS.md + CITATION.cff

---


---

*End of SIMD-Microkernels Copilot Upgrade Instructions — v2.0 target*