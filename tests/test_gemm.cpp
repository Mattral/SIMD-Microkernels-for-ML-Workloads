/**
 * test_gemm.cpp — Numerical correctness tests for the SIMD GEMM kernel.
 *
 * Strategy:
 *   1. Generate random FP32 matrices.
 *   2. Compute reference result with scalar_sgemm (known-correct triple loop).
 *   3. Compute SIMD result with simd_sgemm.
 *   4. Compare element-wise: max |simd - ref| / (|ref| + epsilon) < tolerance.
 *
 * FP32 GEMM accumulation error budget:
 *   For K multiplications + additions, FP32 rounding error is O(K × ε_mach)
 *   where ε_mach ≈ 1.2e-7. For K=512: expected error ≈ 6e-5.
 *   We set tolerance = 1e-3 to allow for FMA re-association with -ffast-math.
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/avx_matmul.hpp"

static void fill_random(float* buf, int n, unsigned seed = 42) {
    // LCG pseudo-random — reproducible, no stdlib dependency
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 200 - 100) * 0.01f;  // [-1, 1]
    }
}

static bool test_gemm_size(int M, int N, int K, double tol = 1e-3) {
    auto A     = make_aligned_array<float>(M * K);
    auto B     = make_aligned_array<float>(K * N);
    auto C_ref = make_aligned_array<float>(M * N);
    auto C_got = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, 1);
    fill_random(B.get(), K * N, 2);

    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, C_ref.get(), N);
    simd_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = std::abs((double)C_got[i] - (double)C_ref[i])
                   / (std::abs((double)C_ref[i]) + 1e-7);
        if (err > max_err) max_err = err;
    }

    bool pass = (max_err < tol);
    printf("  GEMM [%4d×%4d×%4d]  max_rel_err=%.2e  %s\n",
           M, N, K, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int run_gemm_tests() {
    printf("\n── GEMM Correctness Tests ──\n");
    bool all_pass = true;
    all_pass &= test_gemm_size(1,   1,   1);
    all_pass &= test_gemm_size(8,   8,   8);
    all_pass &= test_gemm_size(16,  16,  16);
    all_pass &= test_gemm_size(64,  64,  64);
    all_pass &= test_gemm_size(128, 128, 128);
    all_pass &= test_gemm_size(256, 256, 256);
    all_pass &= test_gemm_size(100, 73,  89);   // non-power-of-2 sizes
    all_pass &= test_gemm_size(33,  17,  5);    // tiny irregular

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
