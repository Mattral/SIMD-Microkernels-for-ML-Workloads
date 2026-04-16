/**
 * test_gemm_packed.cpp — Correctness tests for the packed GEMM implementation.
 *
 * This file validates the BLIS-style packed kernel against a scalar reference
 * for small sizes, non-power-of-two shapes, and tail-case matrix dimensions.
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

static bool test_packed_sgemm_size(int M, int N, int K) {
    auto A = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);

    fill_random(A.get(), M * K, 1);
    fill_random(B.get(), K * N, 2);
    std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
    std::fill(C_got.get(), C_got.get() + M * N, 0.0f);

    naive_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_ref.get(), N);
    simd_ml::gemm::sgemm_packed(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    float max_err = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < M * N; ++i) {
        float err = std::fabs(C_got[i] - C_ref[i]);
        if (err > max_err) {
            max_err = err;
            max_idx = i;
        }
    }

    bool pass = max_err < 1e-4f;
    printf("  packed_sgemm [%4d×%4d×%4d]  max_abs_err=%.3e  %s\n",
           M, N, K, max_err, pass ? "PASS" : "FAIL");
    if (!pass && max_idx >= 0) {
        int row = max_idx / N;
        int col = max_idx % N;
        printf("    max error at row=%d col=%d: got=%.6f ref=%.6f\n",
               row, col, C_got[max_idx], C_ref[max_idx]);
    }
    return pass;
}

int run_gemm_packed_tests() {
    printf("\n── Packed GEMM Correctness Tests ──\n");
    bool all_pass = true;

    const int sizes[] = {1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024};
    for (int size : sizes) {
        all_pass &= test_packed_sgemm_size(size, size, size);
    }

    all_pass &= test_packed_sgemm_size(1, 4096, 4096);
    all_pass &= test_packed_sgemm_size(13, 31, 37);
    all_pass &= test_packed_sgemm_size(32, 7, 17);

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
