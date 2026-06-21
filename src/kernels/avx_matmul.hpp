#pragma once

/**
 * avx_matmul.hpp — Public API for the primary SIMD GEMM microkernel.
 *
 * This header declares the 6×16 AVX2 / 6×32 AVX-512 GEMM implemented in
 * avx_matmul.cpp. For the alternative 8×8 packed GEMM (used by default in
 * Python bindings and the dispatch layer), see gemm/avx2_gemm_packed.hpp.
 *
 * Both implementations follow the Goto/BLIS 5-loop structure with panel
 * packing and register blocking. They serve as complementary references
 * illustrating different micro-kernel tile shapes.
 */

/**
 * simd_sgemm — Tiled AVX2/AVX-512 SGEMM: C = alpha*A*B + beta*C
 *
 * @param M, N, K   Matrix dimensions (A: M×K, B: K×N, C: M×N)
 * @param alpha     Scalar multiplier for A*B
 * @param A, lda    Input A, leading dimension lda ≥ K
 * @param B, ldb    Input B, leading dimension ldb ≥ N
 * @param beta      Scalar multiplier for C (0.0 = overwrite)
 * @param C, ldc    Output C, leading dimension ldc ≥ N
 *
 * All matrices are row-major, float32.
 * 64-byte alignment is recommended for best performance.
 */
void simd_sgemm(int M, int N, int K,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float*       C, int ldc) noexcept;

/**
 * scalar_sgemm — Naïve scalar GEMM (correctness oracle for benchmarks).
 * Uses a simple triple loop with no tiling or vectorization.
 * Sets C = A*B (no alpha/beta scaling) for simplicity.
 */
void scalar_sgemm(int M, int N, int K,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float*       C, int ldc) noexcept;
