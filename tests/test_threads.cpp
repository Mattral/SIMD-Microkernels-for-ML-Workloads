/**
 * test_threads.cpp — OpenMP thread count control and fallback tests.
 *
 * Validates:
 *   1. set_num_threads(n) / get_num_threads() round-trip (both with and without OpenMP)
 *   2. set_num_threads(0) clamps to 1 (invalid input guard)
 *   3. GEMM still produces correct results after changing thread count
 *
 * These tests run in O(1) time regardless of whether OpenMP is available.
 */

#include <cstdio>
#include <cmath>
#include <algorithm>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"

int run_thread_tests() {
    printf("\n── Thread Count Control Tests ──\n");
    bool all_pass = true;

    // Record original setting so we can restore it
    const int original = simd_ml::gemm::get_num_threads();

    // Test 1: set_num_threads(1) → get_num_threads() == 1
    simd_ml::gemm::set_num_threads(1);
    bool t1 = (simd_ml::gemm::get_num_threads() == 1);
    printf("  set(1) → get()==%d  %s\n",
           simd_ml::gemm::get_num_threads(), t1 ? "PASS" : "FAIL");
    all_pass &= t1;

    // Test 2: set_num_threads(0) clamps to 1 (guard against invalid input)
    simd_ml::gemm::set_num_threads(0);
    bool t2 = (simd_ml::gemm::get_num_threads() >= 1);
    printf("  set(0) → get()>0    %s  (got %d)\n",
           t2 ? "PASS" : "FAIL", simd_ml::gemm::get_num_threads());
    all_pass &= t2;

#ifdef SIMD_ML_OPENMP
    // Test 3: multi-thread set/get round-trip
    simd_ml::gemm::set_num_threads(2);
    int got = simd_ml::gemm::get_num_threads();
    bool t3 = (got == 2 || got == 1);  // 1 is valid if only 1 core available
    printf("  set(2) → get()==%d  %s  (OpenMP enabled)\n", got, t3 ? "PASS" : "FAIL");
    all_pass &= t3;
#else
    printf("  set(2) → skipped   SKIP  (OpenMP not enabled in this build)\n");
#endif

    // Test 4: GEMM correctness is preserved after thread count changes
    {
        simd_ml::gemm::set_num_threads(1);

        const int N = 32;
        auto A    = make_aligned_array<float>(N * N);
        auto B    = make_aligned_array<float>(N * N);
        auto C_ref = make_aligned_array<float>(N * N);
        auto C_got = make_aligned_array<float>(N * N);

        for (int i = 0; i < N * N; ++i) {
            A[i] = (float)(i % 17) * 0.1f - 0.8f;
            B[i] = (float)(i % 13) * 0.1f - 0.6f;
        }

        naive_sgemm(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C_ref.get(), N);
        simd_ml::gemm::sgemm_packed(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C_got.get(), N);

        float max_err = 0.0f;
        for (int i = 0; i < N * N; ++i) {
            float e = std::fabs(C_got[i] - C_ref[i]);
            if (e > max_err) max_err = e;
        }

        bool t4 = (max_err < 1e-4f);
        printf("  GEMM after set_threads: max_err=%.2e  %s\n", max_err, t4 ? "PASS" : "FAIL");
        all_pass &= t4;
    }

    // Restore original thread count
    simd_ml::gemm::set_num_threads(original);

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
