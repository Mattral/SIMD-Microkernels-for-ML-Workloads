#include <cstdio>

int run_alignment_tests();
int run_gelu_tests();
int run_gemm_tests();
int run_gemm_packed_tests();

int main() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║  SIMD-ML-Microkernels  ·  Test Suite ║\n");
    printf("╚══════════════════════════════════════╝\n");

    int result = 0;
    result |= run_alignment_tests();
    result |= run_gelu_tests();
    result |= run_gemm_tests();
    result |= run_gemm_packed_tests();

    printf("\n%s\n", result == 0
        ? "═══ ALL TESTS PASSED ═══"
        : "═══ SOME TESTS FAILED ═══");
    return result;
}
