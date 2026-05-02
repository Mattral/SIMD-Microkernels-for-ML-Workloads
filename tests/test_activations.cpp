/**
 * test_activations.cpp — Correctness tests for activation function kernels.
 *
 * Tests: ReLU, SiLU, Softmax, LayerNorm — all validated against scalar
 * reference implementations with tight numerical tolerance.
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/activations/activations.hpp"

static void fill_sweep(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; ++i)
        buf[i] = lo + (hi - lo) * i / std::max(n - 1, 1);
}

static void fill_random(float* buf, int n, unsigned seed = 42) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 200 - 100) * 0.05f;
    }
}

// ─── ReLU tests ──────────────────────────────────────────────────────────────
static bool test_relu(int n, float lo = -5.0f, float hi = 5.0f) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_sweep(inp.get(), n, lo, hi);

    activations::relu_avx2(inp.get(), out.get(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        float ref = inp[i] > 0.0f ? inp[i] : 0.0f;
        double err = std::abs((double)out[i] - (double)ref);
        max_err = std::max(max_err, err);
    }
    bool pass = (max_err == 0.0);
    printf("  ReLU n=%-7d  max_abs_err=%.2e  %s\n", n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── SiLU tests ──────────────────────────────────────────────────────────────
static float ref_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static bool test_silu(int n, float lo = -5.0f, float hi = 5.0f, double tol = 1e-4) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_sweep(inp.get(), n, lo, hi);

    activations::silu_avx2(inp.get(), out.get(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = ref_silu(inp[i]);
        double err = std::abs((double)out[i] - ref) / (std::abs(ref) + 1e-7);
        max_err = std::max(max_err, err);
    }
    bool pass = (max_err < tol);
    printf("  SiLU n=%-7d  max_rel_err=%.2e  %s\n", n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── Softmax tests ────────────────────────────────────────────────────────────
static bool test_softmax(int n, unsigned seed = 1) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_random(inp.get(), n, seed);

    activations::softmax_row_avx2(inp.get(), out.get(), n);

    // Test 1: output sums to 1.0 (within FP32 tolerance)
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += out[i];
    bool sum_ok = std::abs(total - 1.0) < 1e-5;

    // Test 2: all values in [0, 1]
    bool range_ok = true;
    for (int i = 0; i < n; ++i)
        if (out[i] < 0.0f || out[i] > 1.0f + 1e-6f) { range_ok = false; break; }

    // Test 3: argmax of softmax == argmax of input
    int argmax_in  = (int)(std::max_element(inp.get(), inp.get() + n) - inp.get());
    int argmax_out = (int)(std::max_element(out.get(), out.get() + n) - out.get());
    bool argmax_ok = (argmax_in == argmax_out);

    bool pass = sum_ok && range_ok && argmax_ok;
    printf("  Softmax n=%-6d  sum=%.6f  range_ok=%s  argmax_ok=%s  %s\n",
           n, total,
           range_ok  ? "YES" : "NO",
           argmax_ok ? "YES" : "NO",
           pass ? "PASS" : "FAIL");
    return pass;
}

// ─── LayerNorm tests ─────────────────────────────────────────────────────────
static bool test_layer_norm(int n, bool with_gamma_beta, unsigned seed = 5) {
    auto inp   = make_aligned_array<float>(n);
    auto out   = make_aligned_array<float>(n);
    auto gamma = make_aligned_array<float>(n);
    auto beta  = make_aligned_array<float>(n);

    fill_random(inp.get(), n, seed);
    // gamma ~ [0.5, 1.5], beta ~ [-0.1, 0.1]
    for (int i = 0; i < n; ++i) {
        gamma[i] = 1.0f + 0.5f * ((float)(i % 10) / 10.0f - 0.5f);
        beta[i]  = 0.1f * ((float)(i % 7) / 7.0f - 0.5f);
    }

    const float* g = with_gamma_beta ? gamma.get() : nullptr;
    const float* b = with_gamma_beta ? beta.get()  : nullptr;

    activations::layer_norm_avx2(inp.get(), out.get(), n, g, b, 1e-5f);

    // Without gamma/beta: output must have mean ≈ 0 and std ≈ 1
    if (!with_gamma_beta) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += out[i];
        mean /= n;

        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = out[i] - mean;
            var += d * d;
        }
        var /= n;
        double std_dev = std::sqrt(var);

        bool mean_ok = std::abs(mean)         < 1e-4;
        bool std_ok  = std::abs(std_dev - 1.0) < 1e-3;
        bool pass = mean_ok && std_ok;
        printf("  LayerNorm n=%-5d (no γ/β)  mean=%.2e  std=%.5f  %s\n",
               n, mean, std_dev, pass ? "PASS" : "FAIL");
        return pass;
    }

    // With gamma/beta: compute reference scalar and compare
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) { sum += inp[i]; sum2 += (double)inp[i] * inp[i]; }
    double ref_mean = sum / n;
    double ref_var  = sum2 / n - ref_mean * ref_mean;
    double ref_inv_std = 1.0 / std::sqrt(ref_var + 1e-5);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (inp[i] - ref_mean) * ref_inv_std * gamma[i] + beta[i];
        double err = std::abs((double)out[i] - ref) / (std::abs(ref) + 1e-7);
        max_err = std::max(max_err, err);
    }
    bool pass = (max_err < 1e-4);
    printf("  LayerNorm n=%-5d (with γ/β)  max_rel_err=%.2e  %s\n",
           n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── Test runner ─────────────────────────────────────────────────────────────
int run_activation_tests() {
    printf("\n── Activation Correctness Tests ──\n");
    bool all_pass = true;

    printf(" ReLU:\n");
    for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536})
        all_pass &= test_relu(n);

    printf(" SiLU:\n");
    for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536})
        all_pass &= test_silu(n);

    printf(" Softmax:\n");
    for (int n : {1, 2, 8, 16, 64, 512, 4096})
        all_pass &= test_softmax(n);

    printf(" LayerNorm:\n");
    for (int n : {8, 64, 128, 256, 512, 768, 1024})
        all_pass &= test_layer_norm(n, false);  // without affine
    for (int n : {8, 64, 128, 256, 512, 768, 1024})
        all_pass &= test_layer_norm(n, true);   // with gamma/beta

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
