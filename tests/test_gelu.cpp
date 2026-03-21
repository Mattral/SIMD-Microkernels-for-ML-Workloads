/**
 * test_gelu.cpp — Numerical correctness tests for the SIMD GeLU kernel.
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/intrinsic_gelu.hpp"

static bool test_gelu_range(float lo, float hi, int n, double tol = 2e-5) {
    auto input  = make_aligned_array<float>(n);
    auto out_s  = make_aligned_array<float>(n);
    auto out_v  = make_aligned_array<float>(n);

    for (int i = 0; i < n; ++i)
        input[i] = lo + (hi - lo) * i / (n - 1);

    gelu_forward_scalar(input.get(), out_s.get(), n);
    gelu_forward_avx2  (input.get(), out_v.get(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::abs((double)out_v[i] - (double)out_s[i])
                   / (std::abs((double)out_s[i]) + 1e-7);
        if (err > max_err) max_err = err;
    }

    bool pass = (max_err < tol);
    printf("  GeLU [%.1f, %.1f] n=%-7d  max_rel_err=%.2e  %s\n",
           lo, hi, n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int run_gelu_tests() {
    printf("\n── GeLU Correctness Tests ──\n");
    bool all_pass = true;
    all_pass &= test_gelu_range(-5.0f,  5.0f,  1024);
    all_pass &= test_gelu_range(-3.0f,  3.0f,  8192);   // main activation range
    all_pass &= test_gelu_range(-0.5f,  0.5f,  512);    // near-zero
    all_pass &= test_gelu_range( 2.0f,  8.0f,  256);    // saturation region
    all_pass &= test_gelu_range(-8.0f, -2.0f,  256);    // negative saturation
    all_pass &= test_gelu_range(-5.0f,  5.0f,  7);      // odd length (tail path)
    all_pass &= test_gelu_range(-5.0f,  5.0f,  1);      // single element

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
