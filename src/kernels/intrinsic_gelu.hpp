#pragma once

#include <cstddef>

/**
 * gelu_forward_avx2 — Vectorized GeLU evaluation for FP32 values.
 * `input` and `output` may alias for in-place operation.
 */
void gelu_forward_avx2(const float* input,
                       float*       output,
                       std::size_t  n);

/**
 * gelu_forward_scalar — Scalar fallback GeLU implementation for correctness
 * validation and non-AVX2 builds.
 */
void gelu_forward_scalar(const float* input,
                          float*       output,
                          std::size_t  n);
