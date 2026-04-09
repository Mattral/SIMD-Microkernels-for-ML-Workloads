#pragma once

#include <cstddef>

namespace simd_ml {
namespace gemm { void sgemm_packed(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc); }
namespace gemm_ref { void naive_sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc); }
namespace activations { void gelu_avx2(const float*, float*, int); }

namespace dispatch {

using SgemmFn = void(*)(int M, int N, int K,
                        float alpha, const float* A, int lda,
                        const float* B, int ldb,
                        float beta, float* C, int ldc);

using GeluFn = void(*)(const float* input, float* output, int n);

struct KernelRegistry {
    SgemmFn sgemm = nullptr;
    GeluFn gelu = nullptr;
    const char* isa_label = "scalar";
};

const KernelRegistry& get_kernels() noexcept;

// Convenience wrapper exposed to public API
void sgemm(int M, int N, int K,
           float alpha, const float* A, int lda,
           const float* B, int ldb,
           float beta, float* C, int ldc);

const char* detected_isa() noexcept;

} // namespace dispatch
} // namespace simd_ml
