#pragma once

#include <cstddef>

namespace simd_ml {
namespace gemm {

// Reference: Goto, K., & van de Geijn, R. (2008). Anatomy of
// High-Performance Matrix Multiplication. ACM TOMS, 34(3).
// DOI: 10.1145/1356052.1356053

static constexpr int MR = 8;  // register rows (fits 1 YMM per row)
static constexpr int NR = 8;  // register cols
static constexpr int KC = 256;
static constexpr int MC = 128;
static constexpr int NC = 2048;

void pack_b_panel(const float* B, int ldb, int k, int n, float* B_packed);
void pack_a_panel(const float* A, int lda, int m, int k, float* A_packed);
void inner_kernel_8x8(const float* A_packed,
                      const float* B_packed,
                      float* C, int ldc,
                      int k_rem,
                      float alpha);
void sgemm_packed(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

// OpenMP thread control (optional). When OpenMP is enabled at build time
// (`SIMD_ML_OPENMP`) these functions control the number of threads used by
// `sgemm_packed`'s outermost parallel loop.
void set_num_threads(int n);
int get_num_threads();

}  // namespace gemm
}  // namespace simd_ml
