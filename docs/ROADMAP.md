# IntrinsicML — Roadmap

## Legend
✅ Complete and verified   ⚠️ Partial / in progress   ❌ Not started

---

## v0.2 — Correctness & Measurement Foundation ✅
- [✅] Precision test targets compiled without `-ffast-math`
- [✅] Buffer contiguity and dtype validation in Python bindings
- [✅] `tests/test_bindings_edge_cases.py` — rejects non-contiguous NumPy arrays
- [✅] TSC frequency calibration in `main_bench.cpp`
- [✅] CPUID/LFENCE-serialized RDTSC (Intel SDM §4.3)
- [✅] CPU affinity pinning (`sched_setaffinity` on Linux)
- [✅] Real GFLOPS output with hardware utilisation %

---

## v0.3 — True GEMM Microkernel ✅
- [✅] B panel packing (`pack_b_panel`)
- [✅] A panel packing (`pack_a_panel`)
- [✅] AVX2 8×8 micro-kernel with 8 YMM accumulators (`inner_kernel_8x8`)
- [✅] 4× K-loop unrolling in micro-kernel + software prefetch
- [✅] Cache-derived tile constants with L1/L2 derivation comments
- [✅] BLIS-style 5-loop structure (jc / pc / ic / jr / ir)
- [✅] Panel packing buffers allocated once per GEMM call (outside k-loop)
- [✅] Second GEMM implementation (`avx_matmul.cpp` — 6×16 register block)

---

## v0.4 — Activation Kernels ✅
- [✅] GeLU (Cody–Waite exp-based tanh, < 2e-7 absolute error)
- [✅] ReLU (`_mm256_max_ps`, exact)
- [✅] SiLU/Swish (`x·sigmoid(x)` via fast tanh, < 2e-7 error)
- [✅] Softmax (numerically stable, max-subtraction, double accumulation)
- [✅] LayerNorm (3-pass AVX2, double horizontal accumulation, optional γ/β)
- [✅] Shared `simd_math.hpp`: production-grade `fast_exp_avx2` + `tanh_avx2`

---

## v0.5 — Python Integration & Developer Experience ✅
- [✅] pybind11 bindings for all kernels: `sgemm`, `gelu`, `relu`, `silu`, `softmax`, `layer_norm`
- [✅] GIL release during all kernel calls (threading-safe)
- [✅] Python type stubs (`simd_kernels.pyi`) for IDE autocompletion
- [✅] `simd_kernels.build_info()` / `detected_isa()` / `is_aligned()`
- [✅] `set_num_threads` / `get_num_threads` exposed to Python
- [✅] `GEMMConfig` callable configuration object (roadmap §9 DX vision)
- [✅] `isa=` keyword argument on `sgemm()` and `GEMMConfig` (validated, forward-compatible)
- [✅] 100+ Python tests (precision × kernels × shapes, edge cases, error handling)
- [✅] Optional PyTorch cross-validation in test suite

---

## v0.6 — Statistical Benchmarking ✅
- [✅] `bench_stat.cpp`: wall-clock timing, 95% CI, JSON output, CPU pinning
- [✅] `benchmarks/check_regression.py`: CI regression gate (configurable threshold)
- [✅] `bench.yml` weekly cron + automated regression check
- [✅] `docs/BENCHMARKS.md` with roofline analysis and reproducibility instructions
- [✅] OpenBLAS baseline comparison with real measured data (`-DBENCH_OPENBLAS=ON`,
      wired into both `bench` and `bench_stat`; see `docs/BENCHMARKS.md §Positioning vs OpenBLAS`)

---

## v0.7 — Engineering Polish ✅
- [✅] ODR violation resolved (avx_matmul.cpp excluded from precision_lib)
- [✅] AVX-512 dispatch bug fixed (inner_NR=32 misaligned with 6×16 kernel)
- [✅] All build warnings addressed
- [✅] `guardrail.yml`: clang-tidy, pip-audit, codespell, license check
- [✅] `build-and-test.yml`: GCC-12, Clang-15 ASan, macOS ARM scalar path
- [✅] `docs/DESIGN.md` v2: architecture decisions for all kernels

---

## v0.8 — AVX-512 Full Kernel (Planned)

**Status of AVX-512 support today:**
- The `avx_matmul.cpp` file contains a `gemm_micro_6x32_avx512` stub that
  accumulates a single `__m512` (16 floats) per row. This is **incomplete** —
  a correct 6×32 tile requires two `__m512` accumulators per row.
- Runtime dispatch is intentionally **disabled**: `inner_NR` is fixed at 16
  so the proven 6×16 AVX2 kernel runs on all hardware, including AVX-512 CPUs.
  256-bit operations execute at full throughput on AVX-512 cores.
- The `avx2_gemm_packed.cpp` kernel does not yet use AVX-512 at all.

**Planned work:**
- [❌] `gemm_micro_6x32_avx512`: dual-accumulator 6×32 kernel (two `__m512` per row)
- [❌] `gemm_micro_8x16_avx512`: alternative 8×16 layout matching `avx2_gemm_packed` style
- [❌] Re-enable runtime ISA dispatch once the kernel is verified correct
- [❌] Benchmark comparison: AVX2 vs AVX-512 on the same sizes
- [❌] Tile-size re-tuning for Sapphire Rapids (larger L2, different port layout)

---

## v0.9 — Additional Primitives (Planned)
- [❌] Skip panel packing below a size threshold (direct unpacked micro-kernel
      call for small N). Measured impact: IntrinsicML is at 3.7% of OpenBLAS
      at N=64 vs 46–48% at N≥256 — packing overhead dominates small-matrix
      GEMM and is not yet amortised. See `docs/BENCHMARKS.md §Positioning vs
      OpenBLAS` for the measurement that identified this.
- [❌] BF16 GEMM kernel (AVX-512 BF16, relevant for modern LLM inference)
- [❌] FP16 GEMM kernel (F16C extension)
- [✅] Vectorized `exp` via Cody–Waite (`fast_exp_avx2` in `simd_math.hpp`, shared by GeLU/SiLU/Softmax)
- [✅] Vectorized Softmax inner loop using fast exp approximation (`softmax_avx2.cpp` Pass 2)
- [❌] `FlashAttention`-style fused attention microkernel (research target)

---

## v1.0 — NUMA and Multi-socket (Planned)
- [❌] NUMA-aware packing buffer allocation
- [❌] Multi-socket OpenMP with NUMA-local thread-private buffers
- [❌] Benchmark against OpenBLAS multi-threaded on dual-socket systems

---

## GeLU Benchmark Platform Dependency

On Ubuntu 22.04+ with glibc ≥ 2.22, the compiler auto-vectorises loops
containing `erff()` via the `_ZGVdN8v_erff` SVML symbol. This means the scalar
`gelu_forward_scalar` (erff path) and `gelu_forward_avx2` (Cody–Waite tanh)
run at approximately the same throughput on this platform.

**This is not a bug.** On platforms without SVML (Alpine Linux, older glibc,
macOS, embedded targets), `erff` runs scalar and the AVX2 tanh path achieves
6–10× speedup. The AVX2 tanh path also avoids the `erff` function-call overhead
entirely, producing more predictable latency in inference pipelines.

**What this means for benchmarks:** The GeLU column in BENCHMARKS.md shows ~1×
speedup on CI (Ubuntu 22.04 runner). This is accurate for that environment.
On production servers running older glibc or containers based on Alpine, the
speedup will be in the 6–10× range.

---

## Known Honest Limitations (never remove this section)

These limitations are explicitly documented to help users understand the gap
between IntrinsicML and production BLAS libraries.

| Limitation | Impact | Planned fix |
|---|---|---|
| No AVX-512 micro-kernel yet | ~20% throughput gap on AVX-512 hardware | v0.8 |
| No assembly micro-kernel | 5–15% overhead vs hand-tuned assembly | Optional / long-term |
| Fixed tile sizes | Sub-optimal for non-Skylake microarchitectures | v0.9 auto-tune |
| No vectorized `exp()` in Softmax | Softmax 3–5× slower than it could be | v0.9 |
| Single-threaded default | Linear throughput scaling with cores uncaptured | OpenMP flag exists |
| No BF16/FP16 kernels | Modern LLM inference prefers lower precision | v0.9 |
| No NUMA-aware allocation | Throughput degrades on multi-socket systems | v1.0 |
