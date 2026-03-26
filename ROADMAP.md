# Roadmap

## Legend
✅ Complete + verified   ⚠️ Partial   ❌ Not started

---

## v0.2 — Correctness & Measurement Foundation (current target)
- [✅] -ffast-math removed from precision test targets (Fix 1)
- [✅] Buffer contiguity validation in Python bindings (Fix 2)
- [✅] TSC frequency calibration (P0-1)
- [✅] CPUID/lfence-serialized RDTSC (P0-2)
- [✅] CPU affinity pinning (P0-3)
- [✅] Real GFLOPS output with utilization % (P0-4)

## v0.3 — True GEMM Microkernel
- [❌] Matrix B panel packing (P1-1)
- [❌] Matrix A panel packing (P1-1)
- [❌] 4×8 AVX2 inner kernel with explicit accumulators (P1-2)
- [❌] Cache-derived tile size constants with derivation comments (P1-3)
- [❌] BLIS-style 5-loop outer structure (P1-4)
- [❌] Software prefetching in inner loop (P1-5)

## v0.4 — Verified Performance Documentation
- [❌] Real benchmark table with CPU model and measured GFLOPS (P2-1)
- [❌] Cache-arithmetic derivation in README (P2-2)
- [❌] Roofline analysis section in README (P2-3)
- [❌] BENCHMARKS.md with verbatim bench output (P2-4)

## v0.5 — AVX-512 Specialization
- [❌] 4×16 AVX-512 inner kernel (NR=16, __m512 accumulators)
- [❌] Runtime CPU dispatch: detect AVX-512 at startup, dispatch to right kernel
- [❌] Updated tile sizes for AVX-512 (NR=16 changes NC arithmetic)
- [❌] Benchmark comparison: AVX2 vs AVX-512 on same matrix sizes

## v0.6 — Extended Kernels
- [❌] Multithreaded GEMM via OpenMP across outer jc loop
- [❌] NUMA-aware allocation for dual-socket systems
- [❌] BF16 accumulation kernel (relevant for modern ML inference)
- [❌] Benchmark against OpenBLAS (single-threaded, matched build flags)

## Known Honest Limitations (never remove this section)
- No matrix packing until v0.3: kernel is cache-blocked loop, not a microkernel
- Performance numbers are fabricated ranges until v0.4; do not cite them
- GeLU precision test validity uncertain until Fix 1 is applied
- AVX-512 flag is enabled in CMake but no AVX-512-width kernel code exists
