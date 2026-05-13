/**
 * test_gemm_packed.cpp — Correctness tests for avx2_gemm_packed (8×8 micro-kernel).
 *
 * Tests the BLIS-style packed kernel (simd_ml::gemm::sgemm_packed) against the
 * scalar triple-loop reference (naive_sgemm) across:
 *   - Square sizes (1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024)
 *   - Non-square / non-power-of-two sizes (edge/tail handling)
 *   - Alpha/beta scaling paths
 *   - beta=0 overwrite correctness (C must not accumulate prior garbage)
 *
 * Tolerance: 1e-4 absolute. The precision build uses -O2 without -ffast-math,
 * so FP rounding is close to IEEE 754. For larger K the error can reach ~K×ε_mach;
 * the 1e-4 cap is generous enough to cover K=1024 (worst case ~1024×1.2e-7 ≈ 1.2e-4).
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"

static void fill_random(float* buf, int n, unsigned seed = 42) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 200 - 100) * 0.01f;
    }
}

static bool test_sgemm_packed(int M, int N, int K,
                               float alpha = 1.0f, float beta = 0.0f,
                               unsigned seed_c = 0) {
    auto A     = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B     = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);

    fill_random(A.get(), M * K, 1);
    fill_random(B.get(), K * N, 2);
    if (seed_c != 0) {
        fill_random(C_ref.get(), M * N, seed_c);
        std::copy(C_ref.get(), C_ref.get() + M * N, C_got.get());
    } else {
        std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
        std::fill(C_got.get(), C_got.get() + M * N, 0.0f);
    }

    // Reference: naive GEMM (alpha/beta handled inline)
    {
        auto AB = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
        naive_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, AB.get(), N);
        for (int i = 0; i < M * N; ++i)
            C_ref[i] = alpha * AB[i] + beta * C_ref[i];
    }

    simd_ml::gemm::sgemm_packed(M, N, K, alpha, A.get(), K, B.get(), N, beta, C_got.get(), N);

    float max_err = 0.0f;
    int   worst   = -1;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(C_got[i] - C_ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }

    bool pass = (max_err < 1e-4f);
    if (alpha == 1.0f && beta == 0.0f) {
        printf("  [%4d×%4d×%4d]  max_abs=%.2e  %s\n",
               M, N, K, max_err, pass ? "PASS" : "FAIL");
    } else {
        printf("  [%4d×%4d×%4d] α=%.1f β=%.1f  max_abs=%.2e  %s\n",
               M, N, K, alpha, beta, max_err, pass ? "PASS" : "FAIL");
    }
    if (!pass && worst >= 0) {
        printf("    worst at [%d,%d]: got=%.6f  ref=%.6f\n",
               worst / N, worst % N, C_got[worst], C_ref[worst]);
    }
    return pass;
}

static bool test_beta_zero_overwrites_poison() {
    // beta=0 must completely overwrite C, not accumulate into garbage
    const int N = 32;
    auto A = make_aligned_array<float>(N * N);
    auto B = make_aligned_array<float>(N * N);
    auto C = make_aligned_array<float>(N * N);
    fill_random(A.get(), N * N, 20);
    fill_random(B.get(), N * N, 21);
    std::fill(C.get(), C.get() + N * N, 1e30f);  // poison

    simd_ml::gemm::sgemm_packed(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C.get(), N);

    auto C_ref = make_aligned_array<float>(N * N);
    naive_sgemm(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C_ref.get(), N);

    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i)
        max_err = std::max(max_err, std::fabs(C[i] - C_ref[i]));

    bool pass = (max_err < 1e-4f);
    printf("  beta=0 overwrites poison    max_abs=%.2e  %s\n", max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int run_gemm_packed_tests() {
    printf("\n── Packed GEMM Correctness Tests ──\n");
    bool all_pass = true;

    printf(" Standard sizes (α=1, β=0):\n");
    for (int s : {1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024})
        all_pass &= test_sgemm_packed(s, s, s);

    printf(" Non-square edge cases:\n");
    all_pass &= test_sgemm_packed(1,   4096, 4096);
    all_pass &= test_sgemm_packed(13,  31,   37);
    all_pass &= test_sgemm_packed(32,  7,    17);

    printf(" Alpha/beta scaling:\n");
    all_pass &= test_sgemm_packed(32, 32, 32, 2.5f, 0.3f, /*seed_c=*/99);
    all_pass &= test_sgemm_packed(64, 64, 64, 0.5f, 1.0f, /*seed_c=*/77);

    printf(" Edge: beta=0 must overwrite poisoned C:\n");
    all_pass &= test_beta_zero_overwrites_poison();

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
