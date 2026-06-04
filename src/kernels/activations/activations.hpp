#pragma once

#include <cstddef>

namespace activations {

void gelu_avx2(const float* input, float* output, int n);
void relu_avx2(const float* input, float* output, int n);
void silu_avx2(const float* input, float* output, int n);
void softmax_row_avx2(const float* input, float* output, int n);

} // namespace activations
