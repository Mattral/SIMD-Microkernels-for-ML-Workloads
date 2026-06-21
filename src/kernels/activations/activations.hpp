#pragma once

/**
 * activations.hpp — Public API for vectorized activation functions.
 *
 * All functions live in the `activations::` namespace and operate on float32
 * arrays. AVX2 SIMD paths are used when available, with scalar fallbacks.
 *
 * The dispatch layer (kernel_registry.hpp / gemm_dispatcher.cpp) wraps the
 * `activations::gelu_avx2` symbol via its own `simd_ml::activations::` alias
 * declared in kernel_registry.hpp — do NOT add `using` aliases here to avoid
 * declaration conflicts when both headers are included in the same TU.
 */

#include <cstddef>

namespace activations {

/**
 * gelu_avx2 — AVX2-vectorized GeLU using fast tanh rational polynomial.
 * GeLU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
 * Max absolute error vs exact: < 5e-6 over [-5, 5].
 */
void gelu_avx2(const float* input, float* output, int n);

/**
 * relu_avx2 — AVX2-vectorized ReLU: output[i] = max(0, input[i]).
 */
void relu_avx2(const float* input, float* output, int n);

/**
 * silu_avx2 — AVX2-vectorized SiLU/Swish: output[i] = x * sigmoid(x).
 */
void silu_avx2(const float* input, float* output, int n);

/**
 * softmax_row_avx2 — Numerically stable softmax over a single row of n floats.
 */
void softmax_row_avx2(const float* input, float* output, int n);

/**
 * layer_norm_avx2 — Layer normalization over a row of n elements.
 * output[i] = (input[i] - mean) / sqrt(variance + eps) * gamma[i] + beta[i]
 * gamma and beta may be nullptr for the raw normalized output.
 */
void layer_norm_avx2(const float* input, float* output, int n,
                     const float* gamma, const float* beta,
                     float eps = 1e-5f);

} // namespace activations
