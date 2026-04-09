#pragma once

/**
 * naive_gemm.hpp — Scalar GEMM reference implementation.
 *
 * This file provides a simple correctness oracle for packed GEMM and
 * fallback paths. It uses row-major storage and a BLAS-compatible API.
 */

#include <cstddef>

namespace simd_ml {
namespace gemm_ref {

inline void naive_sgemm(int M, int N, int K,
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

} // namespace gemm_ref
} // namespace simd_ml

inline void naive_sgemm(int M, int N, int K,
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) {
    return simd_ml::gemm_ref::naive_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
