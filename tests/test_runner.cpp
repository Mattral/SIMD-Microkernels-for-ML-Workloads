/**
 * test_runner.cpp — Standalone C++ test runner (no framework dependencies).
 *
 * Links against simd_kernels_lib (performance build) and simd_kernels_precision_lib
 * (correctness build, no -ffast-math). See tests/CMakeLists.txt for link order.
 */

#include <cstdio>

// Forward declarations
int run_alignment_tests();
int run_gelu_tests();
int run_gemm_tests();
int run_gemm_packed_tests();
int run_activation_tests();
int run_thread_tests();

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  IntrinsicML  ·  C++ Test Suite          ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    int result = 0;
    result |= run_alignment_tests();
    result |= run_gelu_tests();
    result |= run_gemm_tests();
    result |= run_gemm_packed_tests();
    result |= run_activation_tests();
    result |= run_thread_tests();

    printf("\n%s\n", result == 0
        ? "═══ ALL TESTS PASSED ═══"
        : "═══ SOME TESTS FAILED ═══");
    return result;
}
