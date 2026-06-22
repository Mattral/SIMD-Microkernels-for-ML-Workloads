#pragma once

/**
 * cpuid.hpp — Lightweight CPUID-based CPU feature detection.
 *
 * Used by gemm_dispatcher.cpp to select the best available kernel
 * implementation at process startup (via std::call_once).
 *
 * ─── CPUID leaves used ────────────────────────────────────────────────────────
 * Leaf 1, ECX:
 *   bit 12 — FMA3          (Haswell+)    → required for _mm256_fmadd_ps
 *   bit 20 — SSE4.2        (Nehalem+)    → scalar fallback gate
 *
 * Leaf 7 / sub-leaf 0, EBX:
 *   bit  5 — AVX2          (Haswell+)    → primary SIMD path
 *   bit 16 — AVX-512F      (Skylake-SP+) → detected but not yet used (see ROADMAP.md §v0.8)
 *   bit 17 — AVX-512DQ     (Skylake-SP+) → same
 *
 * ─── Thread safety ────────────────────────────────────────────────────────────
 * CpuFeatures::detect() is called exactly once via std::call_once in
 * gemm_dispatcher.cpp.  The struct itself contains only scalar bools and
 * is safe to read concurrently after initialisation.
 *
 * ─── Non-x86 platforms ───────────────────────────────────────────────────────
 * On non-x86 targets (ARM, RISC-V, etc.) all feature flags remain false
 * and the dispatch layer falls back to naive_sgemm.  Adding ARM NEON/SVE
 * detection is tracked in ROADMAP.md.
 */

#include <cstdint>

namespace simd_ml {
namespace dispatch {

struct CpuFeatures {
    bool has_sse42    = false;
    bool has_fma      = false;
    bool has_avx2     = false;
    bool has_avx512f  = false;
    bool has_avx512dq = false;

    /**
     * detect() — probe CPUID and return a populated CpuFeatures struct.
     * Implemented in gemm_dispatcher.cpp (requires cpuid inline asm).
     * Returns all-false on non-x86 platforms.
     */
    static CpuFeatures detect() noexcept;
};

}  // namespace dispatch
}  // namespace simd_ml
