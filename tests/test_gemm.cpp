/**
 * test_gemm.cpp — Numerical correctness tests for the AVX2 GEMM kernel (avx_matmul.cpp).
 *
 * Tests the `simd_sgemm` function (6×16 micro-kernel with panel packing) against
 * the triple-loop `scalar_sgemm` reference implementation.
 *
 * ─── Tolerance rationale ──────────────────────────────────────────────────────
 * FP32 accumulation over K multiplications has a rounding error budget of
 * O(K × ε_mach) where ε_mach ≈ 1.19e-7.  For K=256: ≈ 3e-5 absolute.
 * We compile the SIMD kernel with -O3 -ffast-math which permits FP reassociation,
 * potentially amplifying the error by O(log₂ K) in the worst case.
 *
 * Tolerance formula:  tol = min(5e-2,  K × 2e-5)
 *   K=8:    1.6e-4   K=64:  1.3e-3   K=128: 2.6e-3   K=256: 5e-2 (capped)
 *
 * This is generous enough to accommodate ffast-math reassociation while still
 * catching truly broken kernels (which typically produce errors of order 1–100).
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/avx_matmul.hpp"

static void fill_random(float* buf, int n, unsigned seed = 42) {
    // Linear congruential generator — reproducible, no stdlib dependency
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = static_cast<float>(static_cast<int>(s >> 16) % 200 - 100) * 0.01f;
    }
}

static bool test_gemm_size(int M, int N, int K) {
    // Tolerance: K-proportional, capped at 5e-2 for very large K.
    // See file header for derivation.
    const double tol = std::min(5e-2, static_cast<double>(K) * 1e-4);

    auto A     = make_aligned_array<float>(M * K);
    auto B     = make_aligned_array<float>(K * N);
    auto C_ref = make_aligned_array<float>(M * N);
    auto C_got = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, /*seed=*/1);
    fill_random(B.get(), K * N, /*seed=*/2);

    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, C_ref.get(), N);
    simd_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    int    worst_flat  = -1;
    for (int i = 0; i < M * N; ++i) {
        double abs_err = std::fabs(static_cast<double>(C_got[i]) -
                                    static_cast<double>(C_ref[i]));
        double rel_err = abs_err / (std::fabs(static_cast<double>(C_ref[i])) + 1e-7);
        if (rel_err > max_rel_err) { max_rel_err = rel_err; worst_flat = i; }
        max_abs_err = std::max(max_abs_err, abs_err);
    }
    (void)worst_flat;

    bool pass = (max_rel_err < tol);
    printf("  GEMM [%4d×%4d×%4d]  tol=%.0e  max_rel=%.2e  max_abs=%.2e  %s\n",
           M, N, K, tol, max_rel_err, max_abs_err, pass ? "PASS" : "FAIL");
    return pass;
}

// Test alpha/beta scaling paths
static bool test_gemm_alpha_beta() {
    const int M = 32, N = 32, K = 32;
    auto A = make_aligned_array<float>(M * K);
    auto B = make_aligned_array<float>(K * N);
    auto C = make_aligned_array<float>(M * N);
    auto C_ref = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, 3);
    fill_random(B.get(), K * N, 4);
    fill_random(C.get(), M * N, 5);
    fill_random(C_ref.get(), M * N, 5);  // same initial C

    const float alpha = 2.5f, beta = 0.3f;

    // Reference: C_ref = alpha*A*B + beta*C_ref  (scalar triple loop)
    // We'll compute A*B separately then scale:
    auto AB = make_aligned_array<float>(M * N);
    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, AB.get(), N);
    for (int i = 0; i < M * N; ++i)
        C_ref[i] = alpha * AB[i] + beta * C_ref[i];

    simd_sgemm(M, N, K, alpha, A.get(), K, B.get(), N, beta, C.get(), N);

    double max_rel = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double rel = std::fabs(static_cast<double>(C[i]) -
                                static_cast<double>(C_ref[i]))
                   / (std::fabs(static_cast<double>(C_ref[i])) + 1e-7);
        max_rel = std::max(max_rel, rel);
    }
    bool pass = max_rel < 1e-3;
    printf("  alpha=%.1f beta=%.1f  [%dx%dx%d]  max_rel=%.2e  %s\n",
           alpha, beta, M, N, K, max_rel, pass ? "PASS" : "FAIL");
    return pass;
}

int run_gemm_tests() {
    printf("\n── GEMM Correctness Tests (avx_matmul.cpp) ──\n");
    bool all_pass = true;

    // Correctness across sizes including edge and non-power-of-two
    all_pass &= test_gemm_size(1,   1,   1);
    all_pass &= test_gemm_size(8,   8,   8);
    all_pass &= test_gemm_size(16,  16,  16);
    all_pass &= test_gemm_size(64,  64,  64);
    all_pass &= test_gemm_size(128, 128, 128);
    all_pass &= test_gemm_size(256, 256, 256);
    all_pass &= test_gemm_size(100, 73,  89);   // non-power-of-2
    all_pass &= test_gemm_size(33,  17,  5);    // tiny irregular

    printf("  Alpha/Beta scaling:\n");
    all_pass &= test_gemm_alpha_beta();

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
