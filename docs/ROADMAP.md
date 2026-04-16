# Roadmap

## Legend
✅ Complete + verified   ⚠️ Partial   ❌ Not started

---

## v0.2 — Correctness & Measurement Foundation (current target)
- [✅] -ffast-math removed from precision test targets (Fix 1)
- [✅] Buffer contiguity validation in Python bindings (Fix 2)
- [✅] `tests/test_bindings_edge_cases.py` added to reject non-contiguous NumPy slices
- [✅] TSC frequency calibration (P0-1)
- [✅] CPUID/lfence-serialized RDTSC (P0-2)
- [✅] CPU affinity pinning (P0-3)
- [✅] Real GFLOPS output with utilization % (P0-4)

## v0.3 — True GEMM Microkernel
- [✅] Matrix B panel packing (P1-1)
- [✅] Matrix A panel packing (P1-1)
- [✅] AVX2 microkernel with explicit register accumulators (P1-2)
- [✅] Cache-derived tile size constants with derivation comments (P1-3)
- [✅] BLIS-style 5-loop outer structure (P1-4)
- [✅] Software prefetching in inner loop (P1-5)

## v0.4 — Verified Performance Documentation
- [✅] Real benchmark table with CPU model and measured GFLOPS (P2-1)
- [✅] Cache-arithmetic derivation in README (P2-2)
- [✅] Roofline analysis section in README (P2-3)
- [✅] BENCHMARKS.md with verbatim bench output (P2-4)

## v0.5 — AVX-512 Specialization
- [⚠️] Basic AVX-512 microkernel implemented (6×32 conservative kernel)
- [✅] Runtime CPU dispatch: detect AVX-512 at runtime and dispatch
- [⚠️] Updated tile-size guidance added, but full AVX-512 re-tuning pending
- [❌] Benchmark comparison: AVX2 vs AVX-512 on same matrix sizes

## v0.6 — Extended Kernels
- [✅] Multithreaded GEMM via OpenMP across outer N-tiles (basic)
- [❌] NUMA-aware allocation for dual-socket systems
- [❌] BF16 accumulation kernel (relevant for modern ML inference)
- [❌] Benchmark against OpenBLAS (single-threaded, matched build flags)

## Known Honest Limitations (never remove this section)
- No AVX-512-width microkernel implemented yet; current code targets AVX2.
 - README contains cache-arithmetic derivation and a roofline/benchmarking methodology section.
- No multithreaded GEMM or NUMA-aware memory allocation yet.
- BF16/low-precision accumulation kernels have not been added.
