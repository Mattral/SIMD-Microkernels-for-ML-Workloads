/**
 * intrinsic_gelu.cpp — Vectorized GELU and Exact Scalar GeLU
 *
 * ─── Mathematical Background ──────────────────────────────────────────────────
 * The Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) is defined as:
 *
 *   GeLU(x) = x · Φ(x)  =  x · 0.5 · [1 + erf(x / √2)]
 *
 * The tanh approximation universally used in BERT/GPT models is:
 *
 *   GeLU_fast(x) ≈ 0.5 · x · [1 + tanh(√(2/π) · (x + 0.044715·x³))]
 *
 * This formulation is what PyTorch's `F.gelu(x, approximate='tanh')` computes.
 * The two formulations differ by up to ~5e-4 in absolute value — both are
 * within the error budget of FP32 model inference.
 *
 * ─── SIMD Strategy (Production-grade exp-based tanh) ─────────────────────────
 * We use the Cody–Waite exp-based tanh from simd_math.hpp rather than a
 * rational polynomial. This gives max absolute error < 2e-7 over all inputs,
 * matching Intel SVML and exceeding rational approximations of similar cost.
 *
 *   tanh(y) = (exp(2y) − 1) / (exp(2y) + 1)
 *
 * exp(z) is computed via:
 *   k = round(z · log₂e),  f = z − k·ln2,  exp(z) = 2^k · poly6(f)
 *
 * Compilation: -O3 -march=native -mfma -mavx2
 */

#include "simd_math.hpp"
#include "intrinsic_gelu.hpp"
#include "cache_alloc.hpp"
#include "activations/activations.hpp"

#include <cmath>
#include <cstddef>

// ─── Compile-time constants ───────────────────────────────────────────────────
static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;   // √(2/π)
static constexpr float GELU_COEFF     = 0.044715f;             // cubic coefficient
static constexpr float INV_SQRT2      = 0.7071067811865476f;   // 1/√2 (for exact erf path)

// ─── Vectorized GeLU via exp-based tanh ──────────────────────────────────────
#ifdef __AVX2__

/**
 * gelu_kernel_avx2 — Compute GeLU(x) for 8 floats using the fast tanh path.
 *
 * GeLU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * Inner argument range for x ∈ [-5, 5]: ≈ [-8.5, 8.5] (→ clamped to ±7.99)
 * Max absolute error vs exact GeLU (erf path): < 5e-4 (inherent formula gap)
 * Max absolute error vs PyTorch tanh-GeLU:     < 5e-7 (our tanh accuracy)
 */
static inline __m256 gelu_kernel_avx2(__m256 x) {
    const __m256 half     = _mm256_set1_ps(0.5f);
    const __m256 one      = _mm256_set1_ps(1.0f);
    const __m256 sqrt2pi  = _mm256_set1_ps(SQRT_2_OVER_PI);
    const __m256 coeff    = _mm256_set1_ps(GELU_COEFF);

    // inner = √(2/π) · (x + 0.044715·x³)
    __m256 x3    = _mm256_mul_ps(_mm256_mul_ps(x, x), x);   // x³
    __m256 inner = _mm256_fmadd_ps(coeff, x3, x);           // x + 0.044715·x³
    inner        = _mm256_mul_ps(sqrt2pi, inner);            // scale

    // tanh(inner) via Cody–Waite exp — < 2e-7 absolute error
    __m256 t = tanh_avx2(inner);

    // 0.5 * x * (1 + tanh)
    return _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t)));
}

#endif  // __AVX2__

// ─── Public: gelu_forward_avx2 ───────────────────────────────────────────────
/**
 * gelu_forward_avx2 — Apply GeLU element-wise (tanh approximation, AVX2 path).
 *
 * This is the fast inference path, equivalent to:
 *   torch.nn.functional.gelu(x, approximate='tanh')
 *
 * Max absolute error vs exact GeLU (erff path): < 5e-4 (formula difference)
 * Max absolute error vs PyTorch tanh GeLU:      < 5e-7 (our tanh accuracy)
 *
 * See gelu_forward_scalar() for the erff-based reference implementation
 * used in correctness tests.
 */
void gelu_forward_avx2(const float* __restrict__ input,
                        float*       __restrict__ output,
                        std::size_t  n) {
#ifdef __AVX2__
    std::size_t i = 0;
    // Main vectorized loop: 8 elements per iteration
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 y = gelu_kernel_avx2(x);
        _mm256_storeu_ps(output + i, y);
    }
    // Scalar tail — handles n % 8 remaining elements
    for (; i < n; ++i) {
        float x     = input[i];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[i]   = 0.5f * x * (1.0f + std::tanh(inner));
    }
#else
    // Non-AVX2 scalar fallback
    for (std::size_t i = 0; i < n; ++i) {
        float x     = input[i];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[i]   = 0.5f * x * (1.0f + std::tanh(inner));
    }
#endif
}

// ─── Reference scalar GeLU (exact erf path — correctness oracle) ─────────────
/**
 * gelu_forward_scalar — Numerically exact GeLU via std::erff.
 *
 * GeLU(x) = 0.5 · x · (1 + erf(x / √2))
 *
 * This is the correctness oracle used in unit tests. It is ~10× slower than
 * the AVX2 tanh path. Note: the tanh-approximate GeLU and erf-exact GeLU
 * differ by up to ~5e-4 in absolute value at x ≈ ±2.7 — this is NOT a bug
 * but an inherent difference between the two mathematical approximations.
 */
void gelu_forward_scalar(const float* input,
                          float*       output,
                          std::size_t  n) {
    for (std::size_t i = 0; i < n; ++i) {
        float x   = input[i];
        output[i] = 0.5f * x * (1.0f + std::erff(x * INV_SQRT2));
    }
}

// ─── activations:: namespace adapter ─────────────────────────────────────────
// The dispatch layer (gemm_dispatcher.cpp) and Python bindings call
// activations::gelu_avx2() — bridge the free function into the namespace here.
namespace activations {

void gelu_avx2(const float* input, float* output, int n) {
    gelu_forward_avx2(input, output, static_cast<std::size_t>(n));
}

} // namespace activations
