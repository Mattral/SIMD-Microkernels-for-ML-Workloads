#pragma once

/**
 * kernel_registry.hpp — Runtime kernel dispatch table.
 *
 * Populated once via std::call_once in gemm_dispatcher.cpp using CPUID.
 * All subsequent kernel calls go through function pointers in KernelRegistry
 * — one indirection, zero branches per dispatch.
 *
 * Kernel selection (in priority order):
 *   AVX2 + FMA → sgemm_packed (8×8 tile) + gelu_avx2 (Cody–Waite exp-tanh)
 *   SSE4.2     → naive_sgemm (scalar)    + gelu scalar fallback in binding
 *   none       → naive_sgemm (scalar)    + gelu scalar fallback in binding
 *
 * Note: even when AVX-512 is detected, we use the AVX2 sgemm_packed because
 * the 6×32 AVX-512 micro-kernel is still incomplete. See ROADMAP.md §v0.8.
 */

namespace simd_ml {

// ─── Forward declarations ─────────────────────────────────────────────────────
// These avoid including the full kernel headers in this lightweight dispatch
// header. Kernel implementations are in avx2_gemm_packed.cpp, gemm_dispatcher.cpp,
// and intrinsic_gelu.cpp.

namespace gemm {
    void sgemm_packed(int M, int N, int K,
                      float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta,
                      float* C, int ldc);
    int  get_num_threads() noexcept;
    void set_num_threads(int n) noexcept;
}  // namespace gemm

namespace gemm_ref {
    void naive_sgemm(int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc);
}  // namespace gemm_ref

namespace dispatch {

// ─── Function pointer types ───────────────────────────────────────────────────

using SgemmFn = void (*)(int M, int N, int K,
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc);

using GeluFn = void (*)(const float* input, float* output, int n);

// ─── Registry struct ──────────────────────────────────────────────────────────

struct KernelRegistry {
    SgemmFn     sgemm     = nullptr;
    GeluFn      gelu      = nullptr;
    const char* isa_label = "scalar";  ///< human-readable ISA name
};

// ─── Registry access ──────────────────────────────────────────────────────────

/**
 * get_kernels — return the populated kernel registry.
 * Initialised exactly once via std::call_once on first call.
 * Thread-safe (C++11 §6.7).
 */
const KernelRegistry& get_kernels() noexcept;

/**
 * sgemm — dispatch to the best available GEMM kernel.
 * Equivalent to get_kernels().sgemm(M, N, K, ...) with a null-check.
 */
void sgemm(int M, int N, int K,
           float alpha,
           const float* A, int lda,
           const float* B, int ldb,
           float beta,
           float* C, int ldc);

/**
 * detected_isa — return a string identifying the selected ISA at runtime.
 * Values: "avx2", "avx512_host_avx2_kernel", "sse42", "scalar".
 */
const char* detected_isa() noexcept;

}  // namespace dispatch
}  // namespace simd_ml
