/**
 * softmax_avx2.cpp — Numerically stable row-wise Softmax.
 *
 * softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 *
 * The max-subtraction step is crucial for numerical stability: without it,
 * exp(x_i) overflows to +inf for large x_i, producing NaN results.
 *
 * Current implementation: scalar with double-precision accumulation.
 * This gives correct results across all tested sizes. An AVX2 fast-exp
 * path (using polynomial approximation of exp) is listed in ROADMAP.md
 * as a planned enhancement.
 *
 * Note on AVX2 vectorized exp: the standard approach is a polynomial
 * approximation of the form exp(x) ≈ 2^(x/ln2) ≈ 2^k * 2^(x - k*ln2)
 * for integer k = round(x/ln2), requiring careful range reduction. This
 * is more complex than the tanh approximation and is deferred to avoid
 * obscuring the core pedagogical structure.
 */

#include "activations.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

namespace activations {

/**
 * softmax_row_avx2 — Numerically stable softmax over a single row of n floats.
 *
 * Two-pass algorithm:
 *   1. Find max(x) to subtract before exp() (prevents overflow).
 *   2. Compute exp(x_i - max), accumulate sum, then divide.
 *
 * Uses double-precision accumulation for the sum to reduce cancellation error
 * when input values span a wide range.
 */
void softmax_row_avx2(const float* input, float* output, int n) {
    if (n <= 0) return;

    // Pass 1: find the maximum value (prevents exp() overflow)
    float maxv = input[0];
    for (int i = 1; i < n; ++i) {
        if (input[i] > maxv) maxv = input[i];
    }

    // Pass 2: exp(x_i - max) and accumulate sum in double precision
    // to reduce rounding error when summing many small values
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        // double-precision exp for more accurate accumulation
        double e  = std::exp(static_cast<double>(input[i] - maxv));
        output[i] = static_cast<float>(e);
        sum      += e;
    }

    // Guard against degenerate case (all inputs = -inf → sum ≈ 0)
    if (sum < 1e-300) {
        // Distribute uniformly rather than produce NaN
        float v = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) output[i] = v;
        return;
    }

    // Normalize: multiply by 1/sum
    float inv_sum = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace activations
