#include <cstddef>
#include <immintrin.h>
#include <cmath>

namespace activations {

// Fast tanh approximation re-used for SiLU's sigmoid via 0.5*(1 + tanh(x/2)).
static inline __m256 tanh_approx_avx2(__m256 y) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 clamp = _mm256_set1_ps(5.0f);
    const __m256 neg_clamp = _mm256_set1_ps(-5.0f);
    y = _mm256_min_ps(y, clamp);
    y = _mm256_max_ps(y, neg_clamp);

    __m256 y2 = _mm256_mul_ps(y, y);
    // Numerator: y * (c1 + c3*y2 + c5*y4)
    __m256 num = _mm256_fmadd_ps(_mm256_set1_ps(0.00533740f), y2, _mm256_set1_ps(-0.16035530f));
    num = _mm256_fmadd_ps(num, y2, one);
    num = _mm256_mul_ps(num, y);
    // Denominator
    __m256 den = _mm256_fmadd_ps(_mm256_set1_ps(0.00587330f), y2, _mm256_set1_ps(0.07985040f));
    den = _mm256_fmadd_ps(den, y2, _mm256_set1_ps(0.48057120f));
    den = _mm256_fmadd_ps(den, y2, one);
    __m256 rcp = _mm256_rcp_ps(den);
    rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(den, rcp, _mm256_set1_ps(2.0f)));
    return _mm256_mul_ps(num, rcp);
}

void silu_avx2(const float* input, float* output, int n) {
#ifdef __AVX2__
    int i = 0;
    const int W = 8;
    for (; i + W <= n; i += W) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 half_x = _mm256_mul_ps(x, _mm256_set1_ps(0.5f));
        __m256 t = tanh_approx_avx2(half_x);
        __m256 sig = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(_mm256_set1_ps(1.0f), t));
        __m256 y = _mm256_mul_ps(x, sig);
        _mm256_storeu_ps(output + i, y);
    }
    for (; i < n; ++i) {
        float x = input[i];
        float sig = 0.5f * (1.0f + std::tanh(0.5f * x));
        output[i] = x * sig;
    }
#else
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sig;
    }
#endif
}

} // namespace activations
