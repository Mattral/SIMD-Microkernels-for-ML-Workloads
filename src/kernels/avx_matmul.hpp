#pragma once

/**
 * avx_matmul.hpp — Public API for the SIMD GEMM microkernel.
 */

/**
 * simd_sgemm — Tiled AVX2/AVX-512 single-precision GEMM.
 *   C = alpha * A * B + beta * C
 *
 * All pointers must be 64-byte aligned (use make_aligned_array or AlignedAllocator).
 */
void simd_sgemm(int M, int N, int K,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float*       C, int ldc);

/**
 * scalar_sgemm — Naive scalar GEMM for comparison / correctness check.
 */
void scalar_sgemm(int M, int N, int K,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float*       C, int ldc);
