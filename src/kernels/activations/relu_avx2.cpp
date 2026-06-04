#include "../../kernels/cache_alloc.hpp"
#include <cstddef>
#include <immintrin.h>

namespace activations {

void relu_avx2(const float* input, float* output, int n) {
#ifdef __AVX2__
    int i = 0;
    const int simd_width = 8;
    __m256 zero = _mm256_setzero_ps();
    for (; i + simd_width <= n; i += simd_width) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output + i, r);
    }
    for (; i < n; ++i) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
#else
    for (int i = 0; i < n; ++i) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
#endif
}

} // namespace activations
