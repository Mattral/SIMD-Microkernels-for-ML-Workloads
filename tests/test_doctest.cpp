#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

int run_alignment_tests();
int run_gelu_tests();
int run_gemm_tests();
int run_gemm_packed_tests();
int run_thread_tests();

TEST_CASE("Alignment correctness") {
    CHECK(run_alignment_tests() == 0);
}

TEST_CASE("GeLU correctness") {
    CHECK(run_gelu_tests() == 0);
}

TEST_CASE("Packed GEMM correctness") {
    CHECK(run_gemm_packed_tests() == 0);
}

TEST_CASE("GEMM correctness") {
    CHECK(run_gemm_tests() == 0);
}

TEST_CASE("Thread count control") {
    CHECK(run_thread_tests() == 0);
}
