/**
 * avx2_gemm_packed.cpp — Packed AVX2 GEMM implementing the Goto/BLIS structure.
 *
 * See DESIGN.md §2 for the cache tiling rationale and §3 for AVX2 register
 * blocking details.
 *
 * This file implements a three-level packing strategy:
 *   1. B panels are packed into contiguous kc×nc blocks.
 *   2. A panels are packed into contiguous mc×kc blocks.
 *   3. The inner kernel computes 8×8 micro-tiles with AVX2/FMA.
 */

#include "avx2_gemm_packed.hpp"
#include "cache_alloc.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>
#ifdef SIMD_ML_OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#else
#  error "Packed GEMM requires AVX2 support"
#endif

namespace simd_ml {
namespace gemm {

static void scale_matrix_c(float* C, int M, int N, int ldc, float beta) {
    if (beta == 1.0f) return;
    for (int i = 0; i < M; ++i) {
        float* row = C + i * ldc;
        if (beta == 0.0f) {
            std::fill(row, row + N, 0.0f);
        } else {
            for (int j = 0; j < N; ++j) {
                row[j] *= beta;
            }
        }
    }
}

void pack_b_panel(const float* B, int ldb, int k, int n, float* B_packed) {
    // Pack the B panel in blocks of NR columns so the inner kernel can load
    // one contiguous __m256 vector per K-iteration.
    for (int jc = 0; jc < n; jc += NR) {
        int nr = std::min(NR, n - jc);
        float* out_block = B_packed + (jc / NR) * k * NR;
        for (int p = 0; p < k; ++p) {
            const float* src = B + p * ldb + jc;
            if (nr == NR) {
                __m256 v = _mm256_loadu_ps(src);
                _mm256_storeu_ps(out_block + p * NR, v);
            } else {
                for (int j = 0; j < nr; ++j) {
                    out_block[p * NR + j] = src[j];
                }
                for (int j = nr; j < NR; ++j) {
                    out_block[p * NR + j] = 0.0f;
                }
            }
        }
    }
}

void pack_a_panel(const float* A, int lda, int m, int k, float* A_packed) {
    // Pack the A panel in row-major order so each 8-row micro-tile can be
    // loaded from contiguous memory with a fixed stride of kc.
    for (int i = 0; i < m; ++i) {
        const float* src = A + i * lda;
        float* dst = A_packed + i * k;
        std::memcpy(dst, src, sizeof(float) * static_cast<std::size_t>(k));
    }
}

static void gemm_scalar_block(int mr, int nr, int kc,
                              const float* A_packed,
                              const float* B_packed,
                              float* C,
                              int ldc,
                              float alpha) {
    // Perform updates in p-major order so the floating-point addition
    // sequence matches the reference naive implementation (summing over
    // K in increasing order). This reduces differences caused by
    // different associativity across KC panels.
    for (int i = 0; i < mr; ++i) {
        float* c_row = C + i * ldc;
        const float* a_row = A_packed + i * kc;
        for (int p = 0; p < kc; ++p) {
            float a_val = a_row[p];
            const float* b_col = B_packed + p * NR;
            for (int j = 0; j < nr; ++j) {
                c_row[j] += alpha * (a_val * b_col[j]);
            }
        }
    }
}

// Accumulate a scalar block into a double-precision accumulation buffer to
// reduce cross-panel rounding error. `C_acc` has dimensions (mr x nr)
// with leading dimension `ldc_acc` (in doubles).
static void gemm_scalar_block_accumulate(int mr, int nr, int kc,
                                         const float* A_packed,
                                         const float* B_packed,
                                         double* C_acc,
                                         int ldc_acc,
                                         double alpha) {
    // Accumulate in p-major order to preserve the same addition sequence
    // across KC panels (but in double precision to reduce rounding).
    for (int i = 0; i < mr; ++i) {
        double* c_row = C_acc + i * ldc_acc;
        const float* a_row = A_packed + i * kc;
        for (int p = 0; p < kc; ++p) {
            double a_val = static_cast<double>(a_row[p]);
            const float* b_col = B_packed + p * NR;
            for (int j = 0; j < nr; ++j) {
                c_row[j] += alpha * a_val * static_cast<double>(b_col[j]);
            }
        }
    }
}

void inner_kernel_8x8(const float* A_packed,
                      const float* B_packed,
                      float* C, int ldc,
                      int k_rem,
                      float alpha) {
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    const float* a0 = A_packed + 0 * k_rem;
    const float* a1 = A_packed + 1 * k_rem;
    const float* a2 = A_packed + 2 * k_rem;
    const float* a3 = A_packed + 3 * k_rem;
    const float* a4 = A_packed + 4 * k_rem;
    const float* a5 = A_packed + 5 * k_rem;
    const float* a6 = A_packed + 6 * k_rem;
    const float* a7 = A_packed + 7 * k_rem;

    const float* b_ptr = B_packed;
    for (int p = 0; p < k_rem; ++p) {
        __m256 b = _mm256_loadu_ps(b_ptr);
        b_ptr += NR;

        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p]), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p]), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p]), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p]), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p]), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p]), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p]), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p]), b, acc7);
    }

    if (alpha != 1.0f) {
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        acc0 = _mm256_mul_ps(acc0, alpha_vec);
        acc1 = _mm256_mul_ps(acc1, alpha_vec);
        acc2 = _mm256_mul_ps(acc2, alpha_vec);
        acc3 = _mm256_mul_ps(acc3, alpha_vec);
        acc4 = _mm256_mul_ps(acc4, alpha_vec);
        acc5 = _mm256_mul_ps(acc5, alpha_vec);
        acc6 = _mm256_mul_ps(acc6, alpha_vec);
        acc7 = _mm256_mul_ps(acc7, alpha_vec);
    }

    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(c0, acc0));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(c1, acc1));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(c2, acc2));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(c3, acc3));
    _mm256_storeu_ps(C + 4 * ldc, _mm256_add_ps(c4, acc4));
    _mm256_storeu_ps(C + 5 * ldc, _mm256_add_ps(c5, acc5));
    _mm256_storeu_ps(C + 6 * ldc, _mm256_add_ps(c6, acc6));
    _mm256_storeu_ps(C + 7 * ldc, _mm256_add_ps(c7, acc7));
}

// Inner kernel variant that accumulates results into a double-precision
// accumulation buffer `C_acc` (ldc_acc in doubles). This avoids repeated
// single-precision updates to the final C matrix across KC panels.
static void inner_kernel_8x8_accumulate_double(const float* A_packed,
                                               const float* B_packed,
                                               double* C_acc, int ldc_acc,
                                               int k_rem,
                                               double alpha) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    const float* a0 = A_packed + 0 * k_rem;
    const float* a1 = A_packed + 1 * k_rem;
    const float* a2 = A_packed + 2 * k_rem;
    const float* a3 = A_packed + 3 * k_rem;
    const float* a4 = A_packed + 4 * k_rem;
    const float* a5 = A_packed + 5 * k_rem;
    const float* a6 = A_packed + 6 * k_rem;
    const float* a7 = A_packed + 7 * k_rem;

    const float* b_ptr = B_packed;
    for (int p = 0; p < k_rem; ++p) {
        __m256 b = _mm256_loadu_ps(b_ptr);
        b_ptr += NR;

        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p]), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p]), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p]), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p]), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p]), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p]), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p]), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p]), b, acc7);
    }

    // Convert each accumulator vector to an array of floats and add into the
    // double accumulation buffer element-wise. This is a small scalar loop
    // over 8 lanes and keeps code simple and correct.
    float tmp[8];

    _mm256_storeu_ps(tmp, acc0);
    for (int i = 0; i < 8; ++i) C_acc[0 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc1);
    for (int i = 0; i < 8; ++i) C_acc[1 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc2);
    for (int i = 0; i < 8; ++i) C_acc[2 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc3);
    for (int i = 0; i < 8; ++i) C_acc[3 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc4);
    for (int i = 0; i < 8; ++i) C_acc[4 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc5);
    for (int i = 0; i < 8; ++i) C_acc[5 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc6);
    for (int i = 0; i < 8; ++i) C_acc[6 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);

    _mm256_storeu_ps(tmp, acc7);
    for (int i = 0; i < 8; ++i) C_acc[7 * ldc_acc + i] += alpha * static_cast<double>(tmp[i]);
}

void sgemm_packed(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    scale_matrix_c(C, M, N, ldc, beta);

    // If OpenMP is enabled, use thread-local packing buffers so each worker
    // thread writes to private scratch space and avoids false sharing.
#ifdef SIMD_ML_OPENMP
    struct ThreadBuffers {
        ThreadBuffers()
            : B_packed(make_aligned_array<float>(static_cast<std::size_t>(KC) *
                                                  static_cast<std::size_t>((NC + NR - 1) / NR) * NR)),
              A_packed(make_aligned_array<float>(static_cast<std::size_t>(MC) *
                                                  static_cast<std::size_t>(KC))) {}

        AlignedUniquePtr<float> B_packed;
        AlignedUniquePtr<float> A_packed;
    };

    thread_local ThreadBuffers buffers;
    float* B_packed_base = buffers.B_packed.get();
    float* A_packed_base = buffers.A_packed.get();

#pragma omp parallel for schedule(dynamic, 1) num_threads(get_num_threads()) if (get_num_threads() > 1)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);
        float* B_packed = B_packed_base;
        float* A_packed = A_packed_base;

        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            pack_b_panel(B + static_cast<std::size_t>(pc) * ldb + jc,
                         ldb,
                         kc,
                         nc,
                         B_packed);

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda,
                             mc,
                             kc,
                             A_packed);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(nc - jr, NR);
                    const float* B_block = B_packed +
                                            static_cast<std::size_t>(jr / NR) * kc * NR;

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(mc - ir, MR);
                        const float* A_block = A_packed + static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc + jc + jr;

                        if (mr == MR && nr == NR) {
                            inner_kernel_8x8(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block(mr, nr, kc, A_block, B_block, C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#else
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            auto B_packed = make_aligned_array<float>(static_cast<std::size_t>(KC) *
                                                      static_cast<std::size_t>((NC + NR - 1) / NR) * NR);
            auto A_packed = make_aligned_array<float>(static_cast<std::size_t>(MC) *
                                                      static_cast<std::size_t>(KC));
            pack_b_panel(B + static_cast<std::size_t>(pc) * ldb + jc,
                         ldb,
                         kc,
                         nc,
                         B_packed.get());

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda,
                             mc,
                             kc,
                             A_packed.get());

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(nc - jr, NR);
                    const float* B_block = B_packed.get() +
                                            static_cast<std::size_t>(jr / NR) * kc * NR;

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(mc - ir, MR);
                        const float* A_block = A_packed.get() + static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc + jc + jr;

                        if (mr == MR && nr == NR) {
                            inner_kernel_8x8(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block(mr, nr, kc, A_block, B_block, C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#endif
}

// Thread control implementation
static std::atomic_int g_num_threads{1};

void set_num_threads(int n) {
    if (n <= 0) n = 1;
    g_num_threads.store(n);
#ifdef SIMD_ML_OPENMP
    omp_set_num_threads(n);
#endif
}

int get_num_threads() {
#ifdef SIMD_ML_OPENMP
    int t = g_num_threads.load();
    if (t <= 0) return omp_get_max_threads();
    return t;
#else
    return 1;
#endif
}

}  // namespace gemm
}  // namespace simd_ml
