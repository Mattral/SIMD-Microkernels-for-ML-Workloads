/**
 * test_doctest.cpp — doctest entry point wrapping all C++ test suites.
 *
 * Compile with -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN so this TU provides main().
 * Each TEST_CASE delegates to the existing run_*_tests() functions to avoid
 * duplicating test logic while gaining doctest's rich failure reporting.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// Forward declarations — implementations in their respective .cpp files
int run_alignment_tests();
int run_gelu_tests();
int run_gemm_tests();
int run_gemm_packed_tests();
int run_activation_tests();
int run_thread_tests();

TEST_CASE("Memory alignment correctness") {
    CHECK(run_alignment_tests() == 0);
}

TEST_CASE("GeLU kernel correctness") {
    CHECK(run_gelu_tests() == 0);
}

TEST_CASE("Packed GEMM correctness (avx2_gemm_packed)") {
    CHECK(run_gemm_packed_tests() == 0);
}

TEST_CASE("SIMD GEMM correctness (avx_matmul)") {
    CHECK(run_gemm_tests() == 0);
}

TEST_CASE("Activation kernels correctness (ReLU/SiLU/Softmax/LayerNorm)") {
    CHECK(run_activation_tests() == 0);
}

TEST_CASE("Thread count control (OpenMP)") {
    CHECK(run_thread_tests() == 0);
}
