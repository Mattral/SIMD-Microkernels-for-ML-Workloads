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
Done

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

**Prompt:**

```
Implement a runtime CPU feature detection and kernel dispatch system.
This is what separates a production kernel library from a demo.

FILE: src/dispatch/cpuid.hpp

Implement a CpuFeatures struct using the CPUID instruction:

struct CpuFeatures {
    bool has_avx2   = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_fma    = false;
    bool has_sse42  = false;

    static CpuFeatures detect() noexcept;
};

Implement CpuFeatures::detect() using:
  - __cpuid_count (GCC/Clang) or __cpuidex (MSVC)
  - Check EBX bit 5 for AVX2, EBX bit 16 for AVX-512F
  - Check ECX bit 12 for FMA (leaf 1)
  - Add a static singleton: const CpuFeatures& CpuFeatures::get()
    using std::call_once for thread-safe initialisation

FILE: src/dispatch/kernel_registry.hpp

Define a function pointer table:

using SgemmFn = void(*)(int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc);

using GeluFn = void(*)(const float* input, float* output, int n);

struct KernelRegistry {
    SgemmFn sgemm;
    GeluFn  gelu;
    const char* isa_label;  // "avx512", "avx2", "sse42", "scalar"
};

const KernelRegistry& get_kernels();
// Returns the optimal registry for the current CPU.
// Called once at program startup via static initialisation.

FILE: src/kernels/gemm/gemm_dispatcher.cpp

Implement get_kernels():
  - If AVX-512F + DQ available: use avx512_gemm_packed, gelu_avx512
  - Else if AVX2 + FMA: use avx2_gemm_packed, gelu_avx2
  - Else if SSE4.2: use sse42_gemm (fallback, unoptimised but correct)
  - Else: use naive_gemm (pure scalar, always correct)

Expose the public API via the dispatcher:
  void sgemm(int M, int N, int K, float alpha, const float* A, int lda,
             const float* B, int ldb, float beta, float* C, int ldc);
  → calls get_kernels().sgemm(...)

This means user code never selects a kernel path manually.

Test in tests/unit/test_dispatcher.cpp:
  - test_dispatcher_selects_avx2_or_better: on a machine with AVX2,
    get_kernels().isa_label must be "avx2" or "avx512"
  - test_dispatcher_result_consistent: sgemm() result must match
    naive_gemm() result for a 32×32 matrix (verifying the dispatch
    doesn't break correctness)
  - test_fallback_scalar_correct: instantiate the scalar path directly
    and verify it produces correct results for a 5×5 matrix

Update pybind11 bindings to expose the ISA label:
  simd_kernels.detected_isa()  # Returns "avx2", "avx512", etc.
```

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

FILE: src/kernels/gemm/avx2_gemm_packed.cpp (modify)

In the outermost loop (nc_block loop) of sgemm_packed(), add:

  #ifdef SIMD_ML_OPENMP
  #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
  #endif
  for (int jc = 0; jc < N; jc += NC) {
    // ... existing loop body
  }

The parallelism must be:
1. Optional: controlled by a compile-time flag SIMD_ML_OPENMP and a runtime
   flag: void set_num_threads(int n_threads);
2. Thread-safe: each thread gets its own A_packed and B_packed buffers
   (allocate per-thread with thread_local storage or a thread pool)
3. Correct: thread_local pack buffers mean no false sharing

Also add:
  int get_num_threads();           // Returns current thread count
  void set_num_threads(int n);     // Sets OpenMP thread count

FILE: CMakeLists.txt (modify)

Add after the existing find_package calls:
  find_package(OpenMP OPTIONAL_COMPONENTS CXX)
  option(SIMD_ML_OPENMP "Enable OpenMP multithreading in GEMM" OFF)
  # OFF by default so single-threaded benchmarks are reproducible

  if(OpenMP_CXX_FOUND AND SIMD_ML_OPENMP)
    target_link_libraries(simd_kernels_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(simd_kernels_lib PUBLIC SIMD_ML_OPENMP)
  endif()

Update pybind11 bindings:
  simd_kernels.set_num_threads(4)   # Exposes OpenMP control to Python
  simd_kernels.get_num_threads()

Tests in tests/unit/test_threading.cpp:
  test_multithreaded_gemm_correct:
    // Run sgemm_packed with 1, 2, 4 threads
    // All must produce bit-identical results to single-threaded
    for threads in [1, 2, 4]:
      set_num_threads(threads)
      result = sgemm(512, 512, 512, ...)
      assert max_abs_diff(result, reference) < 1e-4

  test_thread_count_respected:
    set_num_threads(2)
    assert get_num_threads() == 2
```

---

## BLOCK 5 — Production benchmark harness with statistical rigor

**Create files:** `benchmarks/bench_gemm.cpp`, `benchmarks/run_suite.sh`,
  `benchmarks/results/gemm_results.json`

**Prompt:**

```
Replace the RDTSC-only benchmark harness with a statistically rigorous
benchmarking system. The current harness has no CPU pinning, no frequency
locking, and reports only minimum-of-N. FAANG performance engineers use
proper methodology.

FILE: benchmarks/bench_gemm.cpp

Implement a benchmark that:

1. Measurements:
   - Warm-up: 100 iterations (not timed)
   - Measured iterations: 1000
   - Report: mean, p50, p95, p99, min, max (all in nanoseconds and GFLOPS)
   - Compute GFLOPS: (2 * M * N * K) / (time_ns * 1e-9) / 1e9
     (factor 2 for multiply + add in FMA)

2. Baselines to compare against (via dynamic linking / dlopen if available,
   or via a compile-time optional):
   #ifdef BENCH_OPENBLAS
     #include <cblas.h>
     // Run cblas_sgemm with the same parameters
   #endif
   #ifdef BENCH_EIGEN
     #include <Eigen/Dense>
     // Run Eigen matmul
   #endif
   Always run the naive triple-loop as a mandatory fallback baseline.

3. Matrix sizes to benchmark:
   int sizes[] = {64, 128, 256, 384, 512, 768, 1024, 2048, 4096};
   For each size S: M = N = K = S (square matrices)
   Also: non-square workloads representative of LLM layers:
     {M=1, N=4096, K=4096},   // vector × matrix (decode step)
     {M=128, N=4096, K=4096}, // small batch × matrix (prefill)

4. Output format:
   JSON to benchmarks/results/gemm_results.json:
   {
     "generated_at": "ISO-8601",
     "hardware": {
       "cpu": "...",
       "isa": "avx2|avx512",
       "n_cores": 4,
       "frequency_mhz": 3600,
       "l1_kb": 32, "l2_kb": 256, "l3_mb": 8
     },
     "results": [
       {
         "kernel": "avx2_packed",
         "M": 512, "N": 512, "K": 512,
         "gflops_p50": 45.2,
         "gflops_p95": 44.8,
         "gflops_p99": 43.1,
         "pct_peak": 0.58,
         "vs_openblas_ratio": 0.72
       },
       ...
     ]
   }
   Also print Markdown table to stdout.

5. pct_peak calculation:
   theoretical_peak_gflops = 2.0 * freq_ghz * avx2_fma_lanes * n_threads
   // AVX2 FMA: 2 ops/cycle × 8 FP32/lane = 16 FP32 ops/cycle/core
   // At 3.6GHz, 1 core: 3.6 × 16 = 57.6 GFLOPS/core
   pct_peak = measured_gflops / theoretical_peak_gflops

FILE: benchmarks/run_suite.sh

#!/usr/bin/env bash
set -euo pipefail

echo "=== IntrinsicML Benchmark Suite ==="

# Step 1: Disable CPU frequency scaling (requires sudo; skip gracefully if unavailable)
if command -v cpupower &>/dev/null; then
    echo "Locking CPU frequency to performance governor..."
    sudo cpupower frequency-set -g performance 2>/dev/null || \
        echo "Warning: Could not lock frequency (run as root for reproducible results)"
fi

# Step 2: Pin to a single core using taskset
CORE=0
echo "Pinning to core ${CORE}"
TASKSET_CMD="taskset -c ${CORE}"

# Step 3: Run benchmarks
echo "Running GEMM benchmark..."
${TASKSET_CMD} ./build/bench_gemm --output benchmarks/results/gemm_results.json

echo "Running activation benchmark..."
${TASKSET_CMD} ./build/bench_activations --output benchmarks/results/activation_results.json

echo "=== Benchmark complete. Results in benchmarks/results/ ==="
echo "Methodology: freq-locked (if root), single-core pinned, 1000 trials, p50/p95/p99"

FILE: benchmarks/results/README.md

Document:
1. Hardware used to generate baseline results (CPU, frequency, L-cache sizes)
2. Exact command to regenerate: bash benchmarks/run_suite.sh
3. How to interpret pct_peak (theoretical peak model)
4. Warning: results are hardware-specific; CI results (no freq lock) are
   for regression detection only, not absolute comparison
5. Link to Goto & van de Geijn (2008) for algorithmic background

Commit a real run of gemm_results.json after running the suite.
```

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

FILE: tests/CMakeLists.txt

cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

# AddressSanitizer + UBSanitizer build (separate target)
add_executable(tests_asan
    unit/test_gemm_correctness.cpp
    unit/test_gemm_packed_correctness.cpp
    unit/test_gelu_correctness.cpp
    unit/test_activations_correctness.cpp
    unit/test_dispatcher.cpp
)
target_link_libraries(tests_asan PRIVATE simd_kernels_lib doctest::doctest)
target_compile_options(tests_asan PRIVATE -fsanitize=address,undefined -g -O1)
target_link_options(tests_asan PRIVATE -fsanitize=address,undefined)
# Note: disable -march=native for ASAN builds — it complicates ASAN symbolization

# Normal test executable
add_executable(tests_release
    unit/test_gemm_correctness.cpp
    unit/test_gemm_packed_correctness.cpp
    unit/test_gelu_correctness.cpp
    unit/test_activations_correctness.cpp
    unit/test_dispatcher.cpp
    unit/test_threading.cpp
    precision/test_ulp_error.cpp
)
target_link_libraries(tests_release PRIVATE simd_kernels_lib doctest::doctest)
target_compile_options(tests_release PRIVATE ${SIMD_FLAGS})

add_test(NAME unit_tests_release COMMAND tests_release)
add_test(NAME unit_tests_asan    COMMAND tests_asan)

FILE: tests/unit/test_gemm_correctness.cpp

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "kernels/gemm/avx2_gemm_packed.hpp"
#include "kernels/gemm/naive_gemm.hpp"
#include <vector>
#include <cmath>
#include <cstdlib>

// Helper: random float matrix in [-1, 1]
std::vector<float> rand_matrix(int rows, int cols, unsigned seed = 42) {
    std::srand(seed);
    std::vector<float> m(rows * cols);
    for (auto& v : m) v = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    return m;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        diff = std::max(diff, std::abs(a[i] - b[i]));
    return diff;
}

// Test every size from 1..16 (catches all alignment and tail cases)
TEST_CASE("packed_sgemm correctness small sizes") {
    for (int S = 1; S <= 16; ++S) {
        auto A = rand_matrix(S, S, 1);
        auto B = rand_matrix(S, S, 2);
        std::vector<float> C_our(S*S, 0.0f);
        std::vector<float> C_ref(S*S, 0.0f);
        sgemm_packed(S, S, S, 1.0f, A.data(), S, B.data(), S, 0.0f, C_our.data(), S);
        naive_sgemm(S, S, S, 1.0f, A.data(), S, B.data(), S, 0.0f, C_ref.data(), S);
        float err = max_abs_diff(C_our, C_ref);
        CHECK_MESSAGE(err < 1e-4f,
            "SGEMM error at S=" << S << ": max_abs_diff=" << err);
    }
}

// Test non-square matrices (LLM decode shapes)
TEST_CASE("packed_sgemm LLM decode shape M=1,N=4096,K=4096") {
    int M=1, N=256, K=256;  // Scaled down for CI speed, expand for local
    auto A = rand_matrix(M, K, 3);
    auto B = rand_matrix(K, N, 4);
    std::vector<float> C_our(M*N, 0.0f);
    std::vector<float> C_ref(M*N, 0.0f);
    sgemm_packed(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_our.data(), N);
    naive_sgemm(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
    CHECK(max_abs_diff(C_our, C_ref) < 1e-4f);
}

// Alpha/beta scaling
TEST_CASE("packed_sgemm alpha_beta_scaling") {
    int S = 32;
    auto A = rand_matrix(S, S, 5);
    auto B = rand_matrix(S, S, 6);
    std::vector<float> C_our(S*S, 1.0f);
    std::vector<float> C_ref(S*S, 1.0f);
    sgemm_packed(S, S, S, 2.0f, A.data(), S, B.data(), S, 0.5f, C_our.data(), S);
    naive_sgemm(S, S, S, 2.0f, A.data(), S, B.data(), S, 0.5f, C_ref.data(), S);
    CHECK(max_abs_diff(C_our, C_ref) < 1e-4f);
}

FILE: tests/precision/test_ulp_error.cpp

Test GeLU approximation ULP error:

TEST_CASE("gelu_avx2 max ULP error < 4 in [-5, 5]") {
    // The tanh-based GeLU approximation should have < 4 ULP error
    // vs the exact GeLU computed with std::erf
    const int N = 100000;
    float max_ulp = 0.0f;
    std::vector<float> input(N), output(N);
    for (int i = 0; i < N; ++i) input[i] = -5.0f + 10.0f * i / N;
    gelu_avx2(input.data(), output.data(), N);
    for (int i = 0; i < N; ++i) {
        float exact = 0.5f * input[i] * (1.0f + std::erff(input[i] / std::sqrt(2.0f)));
        float ulp = std::abs(output[i] - exact) /
                    std::numeric_limits<float>::epsilon() / std::abs(exact + 1e-10f);
        max_ulp = std::max(max_ulp, ulp);
    }
    // Document expected ULP budget in test output
    MESSAGE("Max GeLU ULP error: " << max_ulp);
    CHECK(max_ulp < 4.0f);
}
```

---

## BLOCK 7 — CI workflows: build-and-test + bench

**Create files:** `.github/workflows/build-and-test.yml`,
  `.github/workflows/bench.yml`

**Prompt:**

```
Create two GitHub Actions workflows.

FILE: .github/workflows/build-and-test.yml

name: Build and Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-linux-avx2:
    name: Linux / GCC / AVX2
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build libpybind11-dev \
            python3-dev python3-pip clang-15 libopenblas-dev
          pip install pytest numpy

      - name: Configure (Release, AVX2, with OpenBLAS comparison)
        run: |
          cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=g++ \
            -DBENCH_OPENBLAS=ON \
            -DSIMD_ML_OPENMP=ON

      - name: Build
        run: cmake --build build --parallel

      - name: Run C++ unit tests (Release)
        run: ctest --test-dir build -R unit_tests_release --output-on-failure

      - name: Run Python precision tests
        run: |
          pip install -e .
          pytest tests/python/ -v

  build-linux-asan:
    name: Linux / Clang / ASan + UBSan
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: sudo apt-get install -y cmake ninja-build clang-15 libpybind11-dev python3-dev

      - name: Configure (ASAN)
        run: |
          cmake -S . -B build_asan -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_CXX_COMPILER=clang++-15

      - name: Build ASAN target
        run: cmake --build build_asan --target tests_asan

      - name: Run ASAN tests
        run: |
          ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
          UBSAN_OPTIONS=print_stacktrace=1 \
          ctest --test-dir build_asan -R unit_tests_asan --output-on-failure

  build-macos:
    name: macOS / Apple Clang / Scalar fallback
    runs-on: macos-14   # Apple Silicon — no AVX2
    steps:
      - uses: actions/checkout@v4
      - run: brew install cmake pybind11
      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
      - name: Build
        run: cmake --build build --parallel
      - name: Test (scalar fallback path)
        run: ctest --test-dir build --output-on-failure
        # On ARM, dispatcher must select scalar fallback — test this explicitly

FILE: .github/workflows/bench.yml

name: Weekly Benchmark

on:
  schedule:
    - cron: '0 4 * * 0'   # Every Sunday at 4 AM UTC
  workflow_dispatch:        # Also runnable manually

jobs:
  bench:
    name: Benchmark smoke
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Build with OpenBLAS
        run: |
          sudo apt-get install -y cmake libopenblas-dev ninja-build libpybind11-dev python3-dev
          cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBENCH_OPENBLAS=ON
          cmake --build build --parallel

      - name: Run benchmark (no freq lock on GH Actions runner)
        run: |
          # Note: GitHub Actions runners are noisy; these numbers are for
          # regression detection only, not absolute performance claims
          ./build/bench_gemm --sizes "64,128,256" --iterations 200 \
            --output benchmarks/results/ci_gemm_results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: bench-results-${{ github.sha }}
          path: benchmarks/results/ci_gemm_results.json
          retention-days: 90

      - name: Performance regression check
        run: python benchmarks/check_regression.py \
          --baseline benchmarks/results/gemm_results.json \
          --current  benchmarks/results/ci_gemm_results.json \
          --max-regression-pct 15
        # Fail if throughput drops >15% vs committed baseline
```

---

## BLOCK 8 — Dockerfile for reproducible builds

**Create file:** `Dockerfile`

**Prompt:**

```
Create a Dockerfile that builds the complete project from scratch with all
dependencies pinned to exact versions for full reproducibility.

FROM ubuntu:22.04

LABEL maintainer="Min Htet Myet"
LABEL description="IntrinsicML SIMD microkernel development environment"

ARG GCC_VERSION=12
ARG CMAKE_VERSION=3.28.0

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Pin all package versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-${GCC_VERSION}=12.3.0-1ubuntu1~22.04 \
    g++-${GCC_VERSION}=12.3.0-1ubuntu1~22.04 \
    python3.11=3.11.0~rc1-1~22.04 \
    python3.11-dev=3.11.0~rc1-1~22.04 \
    python3-pip \
    ninja-build \
    libopenblas-dev \
    numactl \
    linux-tools-generic \     # for perf
    && rm -rf /var/lib/apt/lists/*

# Install exact CMake version
RUN pip install cmake==${CMAKE_VERSION}

# pybind11 via pip (version-pinned)
RUN pip install pybind11==2.11.1 numpy==1.26.0 pytest==7.4.0

# Set GCC as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100

WORKDIR /workspace
COPY . .

# Build with all features
RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIMD_ML_OPENMP=ON \
    -DBENCH_OPENBLAS=ON \
    && cmake --build build --parallel

# Default: run tests
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]

# To run benchmarks:
# docker run --rm intrinsicml bash benchmarks/run_suite.sh
```

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

## BLOCK 10 — Python API: expand pybind11 bindings + type stubs

**File to modify:** `src/bindings/pybind_entry.cpp`
**Create file:** `simd_kernels.pyi`

**Prompt:**

```
Expand the Python bindings to expose all new kernels and add type stubs
for IDE support.

FILE: src/bindings/pybind_entry.cpp (expand)

Expose:
  simd_kernels.sgemm(A: np.ndarray, B: np.ndarray,
                     alpha: float = 1.0, beta: float = 0.0,
                     C: Optional[np.ndarray] = None) -> np.ndarray
  # Validates: A/B must be float32, C-contiguous, 2D
  # Returns float32 C-contiguous result array
  # Error message must be specific: "A must be float32 C-contiguous 2D array"

  simd_kernels.gelu(x: np.ndarray) -> np.ndarray
  simd_kernels.relu(x: np.ndarray) -> np.ndarray
  simd_kernels.silu(x: np.ndarray) -> np.ndarray
  simd_kernels.softmax(x: np.ndarray, axis: int = -1) -> np.ndarray

  simd_kernels.detected_isa() -> str  # "avx512", "avx2", "sse42", "scalar"
  simd_kernels.build_info() -> dict   # existing, but expand
  simd_kernels.set_num_threads(n: int) -> None
  simd_kernels.get_num_threads() -> int

Array input contract (enforce via pybind11):
  - Must be float32
  - Must be C-contiguous (F-contiguous triggers a copy with a warning)
  - Must be non-empty
  - For sgemm: A.shape[1] must == B.shape[0]

FILE: simd_kernels.pyi  (Python type stub for IDE/mypy support)

from typing import Optional
import numpy as np
from numpy.typing import NDArray

def sgemm(A: NDArray[np.float32], B: NDArray[np.float32],
          alpha: float = ..., beta: float = ...,
          C: Optional[NDArray[np.float32]] = ...) -> NDArray[np.float32]: ...

def gelu(x: NDArray[np.float32]) -> NDArray[np.float32]: ...
def relu(x: NDArray[np.float32]) -> NDArray[np.float32]: ...
def silu(x: NDArray[np.float32]) -> NDArray[np.float32]: ...
def softmax(x: NDArray[np.float32], axis: int = ...) -> NDArray[np.float32]: ...
def detected_isa() -> str: ...
def build_info() -> dict: ...
def set_num_threads(n: int) -> None: ...
def get_num_threads() -> int: ...

Update tests/python/test_precision.py to add:
  - test_sgemm_agrees_numpy: simd_kernels.sgemm(A, B) must agree with
    np.float32(np.matmul(A.astype(float), B.astype(float))) within 1e-4
  - test_softmax_agrees_scipy: simd_kernels.softmax(x) must agree with
    scipy.special.softmax(x) within 1e-5
  - test_type_error_on_float64: passing float64 array must raise TypeError
  - test_contiguous_check: F-contiguous input is handled without crash
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
- [ ] BLOCK 1: Packed GEMM (Goto algorithm) — avx2_gemm_packed.cpp
- [ ] BLOCK 2: Runtime ISA dispatcher (cpuid + kernel_registry)
- [ ] BLOCK 5: Production benchmark harness (GFLOPS + OpenBLAS comparison)
- [ ] Commit actual benchmark results to benchmarks/results/

**P1 — Completeness**
- [ ] BLOCK 3: ReLU, SiLU, Softmax kernels
- [ ] BLOCK 4: OpenMP multithreading
- [ ] BLOCK 6: C++ unit tests with doctest + ASan/UBSan
- [ ] BLOCK 7: CI workflows (build-and-test + bench)

**P2 — Polish**
- [ ] BLOCK 8: Dockerfile
- [ ] BLOCK 9: DESIGN.md
- [ ] BLOCK 10: Expanded pybind11 bindings + .pyi type stubs
- [ ] BLOCK 11: README overhaul + BENCHMARKS.md + CITATION.cff

---

## ANSWERED QUESTIONS — SUMMARY TABLE

| Question | Answer | Rationale |
|---|---|---|
| 100% test coverage? | 95% on kernel source, 100% on public API | Kernel code needs ULP/precision testing, not branch padding |
| CI? | Yes — 3 jobs: AVX2, ASan, macOS ARM scalar fallback | Catches UB, platform-specific bugs, and performance regressions |
| Docker? | Yes — one Dockerfile, pinned GCC + CMake | Reproducible builds without local toolchain setup |
| Rust? | No rewrite — optional thin PyO3 wrapper only | C++ intrinsics are the right tool; Rust SIMD is less mature for this use case |
| How to evaluate? | GFLOPS p50/p95/p99 vs OpenBLAS, % of theoretical peak | Industry-standard metric, hardware-independent, comparable across machines |
| How to document? | DESIGN.md (roofline model, cache tiling, register blocking) + source file headers referencing DESIGN.md sections | Signals systems depth to compiler/hardware team interviewers |
| How to be top-tier? | Implement packed GEMM, commit real OpenBLAS comparison numbers, add runtime ISA dispatch | These three moves transform it from "demo" to "credible reference" |

---

*End of SIMD-Microkernels Copilot Upgrade Instructions — v2.0 target*