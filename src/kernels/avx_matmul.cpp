/**
 * avx_matmul.cpp — Hand-Vectorized AVX2 / AVX-512 GEMM Microkernel
 *
 * Computes: C = A × B   (all FP32, row-major layout)
 *   A: M×K,  B: K×N,  C: M×N
 *
 * ─── Micro-Architectural Strategy ────────────────────────────────────────────
 *
 * 1. LOOP TILING / CACHE BLOCKING
 *    L1d cache on Intel Core (Alder Lake P-core): 48 KB
 *    L2 cache: 1.25 MB
 *    Tile sizes chosen so three sub-matrices (A tile, B tile, C accumulator)
 *    fit concurrently in L1d:
 *      MC=64, KC=256, NC=64  →  (64×256 + 256×64 + 64×64) × 4B = ~196 KB  [L2 resident]
 *    Inner micro-kernel: MR=6 rows × NR=16 cols (AVX2) or NR=32 (AVX-512)
 *    fits accumulator registers entirely inside the 16 YMM / 32 ZMM file.
 *
 * 2. REGISTER BLOCKING
 *    AVX2 path: 6×2 register block (6 __m256 accumulators × 2 AVX lanes wide)
 *    = 12 YMM registers for C, 1 for A broadcast, 2 for B panels → 15 / 16 YMM used.
 *    This leaves the CPU renamer with no spill pressure.
 *
 * 3. FUSED MULTIPLY-ADD (FMA)
 *    _mm256_fmadd_ps: 1 instruction, 2 FP operations, 0.5 cycle throughput (AVX2).
 *    Peak FP throughput per core (AVX2, 3.5 GHz): 3.5e9 × 2 FMA/cy × 8 FP/FMA = 56 GFLOPS.
 *
 * 4. SOFTWARE PREFETCHING
 *    Prefetch B panel 4 iterations ahead into L1 to hide DRAM/L2 latency.
 *
 * 5. MANUAL LOOP UNROLLING (×4 in K-loop inner body)
 *    Reduces loop-overhead branch instructions and exposes the scheduler to
 *    a longer instruction window for out-of-order execution.
 *
 * Compilation flags expected:
 *   GCC/Clang: -O3 -march=native -mfma   (or -mavx512f -mavx512dq for AVX-512 path)
 *   MSVC:      /O2 /arch:AVX2
 */

#include "cache_alloc.hpp"
#include "avx_matmul.hpp"

#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#  include <immintrin.h>
#  define SIMD_WIDTH 16          // 16 × float32 in __m512
#  define SIMD_SUFFIX "AVX-512"
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define SIMD_WIDTH 8           // 8 × float32 in __m256
#  define SIMD_SUFFIX "AVX2"
#else
#  error "This kernel requires at least AVX2. Compile with -mavx2 -mfma."
#endif

// ─── Tiling constants (tuned for 48 KB L1d, 1.25 MB L2) ─────────────────────
static constexpr int MC = 64;    // M-dimension tile (rows of A / C)
static constexpr int KC = 256;   // K-dimension tile (cols of A / rows of B)
static constexpr int NC = 64;    // N-dimension tile (cols of B / C)

// Micro-kernel register block (must satisfy MR*NR*4 << YMM register file)
static constexpr int MR = 6;     // rows handled per micro-kernel call
static constexpr int NR = 2 * SIMD_WIDTH;  // cols: 2 AVX vectors wide

// ─── AVX2 6×16 Micro-kernel ───────────────────────────────────────────────────
#ifdef __AVX2__
/**
 * gemm_micro_6x16_avx2 — computes a 6×16 sub-block of C += A_panel × B_panel
 *
 * @param kc   — depth of the K-panel (≤ KC)
 * @param A    — pointer into packed A panel, shape [MR × kc], row-major
 * @param B    — pointer into packed B panel, shape [kc × NR], row-major
 * @param C    — pointer into output C block, row stride = ldc
 * @param ldc  — leading dimension (column stride) of C
 */
static void gemm_micro_6x16_avx2(int kc,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float*       __restrict__ C,
                                   int ldc) {
    // Accumulator registers: 6 rows × 2 AVX vectors (16 floats) = 12 YMM
    __m256 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m256 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    // Load existing C values (accumulate into C, not overwrite)
    c0_lo = _mm256_loadu_ps(C + 0*ldc + 0);  c0_hi = _mm256_loadu_ps(C + 0*ldc + 8);
    c1_lo = _mm256_loadu_ps(C + 1*ldc + 0);  c1_hi = _mm256_loadu_ps(C + 1*ldc + 8);
    c2_lo = _mm256_loadu_ps(C + 2*ldc + 0);  c2_hi = _mm256_loadu_ps(C + 2*ldc + 8);
    c3_lo = _mm256_loadu_ps(C + 3*ldc + 0);  c3_hi = _mm256_loadu_ps(C + 3*ldc + 8);
    c4_lo = _mm256_loadu_ps(C + 4*ldc + 0);  c4_hi = _mm256_loadu_ps(C + 4*ldc + 8);
    c5_lo = _mm256_loadu_ps(C + 5*ldc + 0);  c5_hi = _mm256_loadu_ps(C + 5*ldc + 8);

    const float* a_ptr = A;
    const float* b_ptr = B;

    // K-loop unrolled ×4 to reduce branch overhead and widen ILP window
    int k = 0;
    for (; k <= kc - 4; k += 4) {
        // ── k+0 ──────────────────────────────────────────────────────────────
        __m256 b_lo, b_hi, a_bcast;

        prefetch_l1(b_ptr + 32);        // pull next B row into L1

        b_lo = _mm256_load_ps(b_ptr);
        b_hi = _mm256_load_ps(b_ptr + 8);
        b_ptr += 16;

        a_bcast = _mm256_set1_ps(a_ptr[0]);
        c0_lo = _mm256_fmadd_ps(a_bcast, b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(a_bcast, b_hi, c0_hi);

        a_bcast = _mm256_set1_ps(a_ptr[1]);
        c1_lo = _mm256_fmadd_ps(a_bcast, b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(a_bcast, b_hi, c1_hi);

        a_bcast = _mm256_set1_ps(a_ptr[2]);
        c2_lo = _mm256_fmadd_ps(a_bcast, b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(a_bcast, b_hi, c2_hi);

        a_bcast = _mm256_set1_ps(a_ptr[3]);
        c3_lo = _mm256_fmadd_ps(a_bcast, b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(a_bcast, b_hi, c3_hi);

        a_bcast = _mm256_set1_ps(a_ptr[4]);
        c4_lo = _mm256_fmadd_ps(a_bcast, b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(a_bcast, b_hi, c4_hi);

        a_bcast = _mm256_set1_ps(a_ptr[5]);
        c5_lo = _mm256_fmadd_ps(a_bcast, b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(a_bcast, b_hi, c5_hi);
        a_ptr += MR;

        // ── k+1 ──────────────────────────────────────────────────────────────
        prefetch_l1(b_ptr + 32);

        b_lo = _mm256_load_ps(b_ptr);
        b_hi = _mm256_load_ps(b_ptr + 8);
        b_ptr += 16;

        a_bcast = _mm256_set1_ps(a_ptr[0]);
        c0_lo = _mm256_fmadd_ps(a_bcast, b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(a_bcast, b_hi, c0_hi);

        a_bcast = _mm256_set1_ps(a_ptr[1]);
        c1_lo = _mm256_fmadd_ps(a_bcast, b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(a_bcast, b_hi, c1_hi);

        a_bcast = _mm256_set1_ps(a_ptr[2]);
        c2_lo = _mm256_fmadd_ps(a_bcast, b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(a_bcast, b_hi, c2_hi);

        a_bcast = _mm256_set1_ps(a_ptr[3]);
        c3_lo = _mm256_fmadd_ps(a_bcast, b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(a_bcast, b_hi, c3_hi);

        a_bcast = _mm256_set1_ps(a_ptr[4]);
        c4_lo = _mm256_fmadd_ps(a_bcast, b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(a_bcast, b_hi, c4_hi);

        a_bcast = _mm256_set1_ps(a_ptr[5]);
        c5_lo = _mm256_fmadd_ps(a_bcast, b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(a_bcast, b_hi, c5_hi);
        a_ptr += MR;

        // ── k+2 ──────────────────────────────────────────────────────────────
        b_lo = _mm256_load_ps(b_ptr);
        b_hi = _mm256_load_ps(b_ptr + 8);
        b_ptr += 16;

        a_bcast = _mm256_set1_ps(a_ptr[0]);
        c0_lo = _mm256_fmadd_ps(a_bcast, b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(a_bcast, b_hi, c0_hi);

        a_bcast = _mm256_set1_ps(a_ptr[1]);
        c1_lo = _mm256_fmadd_ps(a_bcast, b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(a_bcast, b_hi, c1_hi);

        a_bcast = _mm256_set1_ps(a_ptr[2]);
        c2_lo = _mm256_fmadd_ps(a_bcast, b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(a_bcast, b_hi, c2_hi);

        a_bcast = _mm256_set1_ps(a_ptr[3]);
        c3_lo = _mm256_fmadd_ps(a_bcast, b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(a_bcast, b_hi, c3_hi);

        a_bcast = _mm256_set1_ps(a_ptr[4]);
        c4_lo = _mm256_fmadd_ps(a_bcast, b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(a_bcast, b_hi, c4_hi);

        a_bcast = _mm256_set1_ps(a_ptr[5]);
        c5_lo = _mm256_fmadd_ps(a_bcast, b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(a_bcast, b_hi, c5_hi);
        a_ptr += MR;

        // ── k+3 ──────────────────────────────────────────────────────────────
        b_lo = _mm256_load_ps(b_ptr);
        b_hi = _mm256_load_ps(b_ptr + 8);
        b_ptr += 16;

        a_bcast = _mm256_set1_ps(a_ptr[0]);
        c0_lo = _mm256_fmadd_ps(a_bcast, b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(a_bcast, b_hi, c0_hi);

        a_bcast = _mm256_set1_ps(a_ptr[1]);
        c1_lo = _mm256_fmadd_ps(a_bcast, b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(a_bcast, b_hi, c1_hi);

        a_bcast = _mm256_set1_ps(a_ptr[2]);
        c2_lo = _mm256_fmadd_ps(a_bcast, b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(a_bcast, b_hi, c2_hi);

        a_bcast = _mm256_set1_ps(a_ptr[3]);
        c3_lo = _mm256_fmadd_ps(a_bcast, b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(a_bcast, b_hi, c3_hi);

        a_bcast = _mm256_set1_ps(a_ptr[4]);
        c4_lo = _mm256_fmadd_ps(a_bcast, b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(a_bcast, b_hi, c4_hi);

        a_bcast = _mm256_set1_ps(a_ptr[5]);
        c5_lo = _mm256_fmadd_ps(a_bcast, b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(a_bcast, b_hi, c5_hi);
        a_ptr += MR;
    }
    // Scalar tail for remaining k iterations
    for (; k < kc; ++k) {
        __m256 b_lo = _mm256_load_ps(b_ptr);
        __m256 b_hi = _mm256_load_ps(b_ptr + 8);
        b_ptr += 16;

        c0_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[0]), b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[0]), b_hi, c0_hi);
        c1_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[1]), b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[1]), b_hi, c1_hi);
        c2_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[2]), b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[2]), b_hi, c2_hi);
        c3_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[3]), b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[3]), b_hi, c3_hi);
        c4_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[4]), b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[4]), b_hi, c4_hi);
        c5_lo = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[5]), b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(_mm256_set1_ps(a_ptr[5]), b_hi, c5_hi);
        a_ptr += MR;
    }

    // Write-back accumulator to C
    _mm256_storeu_ps(C + 0*ldc + 0, c0_lo); _mm256_storeu_ps(C + 0*ldc + 8, c0_hi);
    _mm256_storeu_ps(C + 1*ldc + 0, c1_lo); _mm256_storeu_ps(C + 1*ldc + 8, c1_hi);
    _mm256_storeu_ps(C + 2*ldc + 0, c2_lo); _mm256_storeu_ps(C + 2*ldc + 8, c2_hi);
    _mm256_storeu_ps(C + 3*ldc + 0, c3_lo); _mm256_storeu_ps(C + 3*ldc + 8, c3_hi);
    _mm256_storeu_ps(C + 4*ldc + 0, c4_lo); _mm256_storeu_ps(C + 4*ldc + 8, c4_hi);
    _mm256_storeu_ps(C + 5*ldc + 0, c5_lo); _mm256_storeu_ps(C + 5*ldc + 8, c5_hi);
}
#endif  // __AVX2__

// ─── Pack helpers (improve B panel access pattern) ───────────────────────────
/**
 * pack_B — reorder B[k0..k0+kc, n0..n0+nc] into a contiguous K-major panel.
 * This converts the strided column access of B into sequential reads, turning
 * cache-thrashing random access into streaming prefetch-friendly access.
 */
static void pack_B(const float* B, int ldb,
                   int kc, int nc,
                   float* __restrict__ B_packed) {
    const int num_blocks = (nc + NR - 1) / NR;
    for (int block = 0; block < num_blocks; ++block) {
        int jb = block * NR;
        int actual_nr = std::min(NR, nc - jb);
        float* dst_base = B_packed + block * kc * NR;

        for (int k = 0; k < kc; ++k) {
            const float* src = B + k * ldb + jb;
            float* dst = dst_base + k * NR;

            for (int n = 0; n < actual_nr; ++n) {
                dst[n] = src[n];
            }
            for (int n = actual_nr; n < NR; ++n) {
                dst[n] = 0.0f;
            }
        }
    }
}

/**
 * pack_A — reorder A[m0..m0+mr, k0..k0+kc] into column-major micro-panel.
 * Gives the inner K-loop sequential access for each row broadcast.
 */
static void pack_A(const float* A, int lda,
                   int mr, int kc,
                   float* __restrict__ A_packed,
                   float alpha = 1.0f) {
    int num_blocks = (mr + MR - 1) / MR;
    for (int block = 0; block < num_blocks; ++block) {
        int rows = std::min(MR, mr - block * MR);
        float* dst_base = A_packed + block * kc * MR;
        const float* A_block = A + block * MR * lda;

        for (int k = 0; k < kc; ++k) {
            const float* src = A_block + k;
            float* dst = dst_base + k * MR;

            for (int m = 0; m < rows; ++m) {
                dst[m] = alpha * src[m * lda];
            }
            for (int m = rows; m < MR; ++m) {
                dst[m] = 0.0f;
            }
        }
    }
}

// ─── Public API: sgemm ────────────────────────────────────────────────────────
/**
 * simd_sgemm — Compute C = alpha*A*B + beta*C  (single-precision)
 *
 * @param M, N, K   — matrix dimensions
 * @param alpha     — scalar multiplier for A*B
 * @param A         — input matrix A [M×K], row-major, 64-byte aligned
 * @param lda       — leading dimension of A (≥ K)
 * @param B         — input matrix B [K×N], row-major, 64-byte aligned
 * @param ldb       — leading dimension of B (≥ N)
 * @param beta      — scalar multiplier for C (0.0 = overwrite)
 * @param C         — output matrix C [M×N], row-major, 64-byte aligned
 * @param ldc       — leading dimension of C (≥ N)
 */
void simd_sgemm(int M, int N, int K,
                float alpha,
                const float* __restrict__ A, int lda,
                const float* __restrict__ B, int ldb,
                float beta,
                float*       __restrict__ C, int ldc) {
    // Apply beta scaling to C upfront
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i)
            std::memset(C + i * ldc, 0, N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
    }

    // Number of N-tiles
    int num_j_blocks = (N + NC - 1) / NC;

    // Parallelize across N-tiles (outermost loop) with OpenMP if available.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (int jb = 0; jb < num_j_blocks; ++jb) {
#else
    for (int jb = 0; jb < num_j_blocks; ++jb) {
#endif
        int j = jb * NC;
        int nc = std::min(NC, N - j);

        // Per-thread packing buffers to avoid races
        auto A_pack = make_aligned_array<float>(((MC + MR - 1) / MR) * KC * MR);
        auto B_pack = make_aligned_array<float>(((NC + NR - 1) / NR) * KC * NR);

        for (int k = 0; k < K; k += KC) {
            int kc = std::min(KC, K - k);

            // Pack B panel into L2-friendly layout
            pack_B(B + k * ldb + j, ldb, kc, nc, B_pack.get());

            for (int i = 0; i < M; i += MC) {
                int mc = std::min(MC, M - i);

                // Pack A panel for this thread
                pack_A(A + i * lda + k, lda, mc, kc, A_pack.get(), alpha);

                for (int ii = 0; ii < mc; ii += MR) {
                    int mr = std::min(MR, mc - ii);
                    const float* A_micro = A_pack.get() + (ii / MR) * (kc * MR);
                    for (int jj = 0; jj < nc; jj += NR) {
                        int nr = std::min(NR, nc - jj);
                        if (mr == MR && nr == NR) {
                            int block = jj / NR;
                            float* B_block = B_pack.get() + block * kc * NR;
                            // Runtime dispatch: prefer AVX-512 if available at runtime
#if defined(__AVX512F__)
                            bool have_avx512 = false;
#if defined(__GNUC__) || defined(__clang__)
                            have_avx512 = __builtin_cpu_supports("avx512f");
#endif
                            if (have_avx512) {
                                // AVX-512 optimized microkernel (if compiled with AVX-512)
                                gemm_micro_6x32_avx512(
                                    kc,
                                    A_micro,
                                    B_block,
                                    C + (i + ii) * ldc + (j + jj),
                                    ldc
                                );
                            } else
#endif
                            {
#ifdef __AVX2__
                                gemm_micro_6x16_avx2(
                                    kc,
                                    A_micro,
                                    B_block,
                                    C + (i + ii) * ldc + (j + jj),
                                    ldc
                                );
#else
                                scalar_sgemm(MR, NR, kc,
                                             A + (i + ii) * lda + k, lda,
                                             B + k * ldb + (j + jj), ldb,
                                             C + (i + ii) * ldc + (j + jj), ldc);
#endif
                            }
                        } else {
                            scalar_sgemm(mr, nr, kc,
                                         A + (i + ii) * lda + k, lda,
                                         B + k * ldb + (j + jj), ldb,
                                         C + (i + ii) * ldc + (j + jj), ldc);
                        }
                    }
                }
            }
        }
    }


#if defined(__AVX512F__)
/**
 * Simple AVX-512 microkernel: 6×32 blocking using __m512 accumulators.
 * This is a straightforward port of the AVX2 kernel for machines compiled
 * with AVX-512 support. It is intentionally conservative to avoid register
 * pressure while demonstrating a 512-bit path and a runtime dispatch.
 */
static void gemm_micro_6x32_avx512(int kc,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float*       __restrict__ C,
                                   int ldc) {
    __m512 c0, c1, c2, c3, c4, c5;
    c0 = _mm512_loadu_ps(C + 0*ldc);
    c1 = _mm512_loadu_ps(C + 1*ldc);
    c2 = _mm512_loadu_ps(C + 2*ldc);
    c3 = _mm512_loadu_ps(C + 3*ldc);
    c4 = _mm512_loadu_ps(C + 4*ldc);
    c5 = _mm512_loadu_ps(C + 5*ldc);

    const float* a_ptr = A;
    const float* b_ptr = B;

    for (int k = 0; k < kc; ++k) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 32;

        __m512 a0 = _mm512_set1_ps(a_ptr[0]); c0 = _mm512_fmadd_ps(a0, b, c0);
        __m512 a1 = _mm512_set1_ps(a_ptr[1]); c1 = _mm512_fmadd_ps(a1, b, c1);
        __m512 a2 = _mm512_set1_ps(a_ptr[2]); c2 = _mm512_fmadd_ps(a2, b, c2);
        __m512 a3 = _mm512_set1_ps(a_ptr[3]); c3 = _mm512_fmadd_ps(a3, b, c3);
        __m512 a4 = _mm512_set1_ps(a_ptr[4]); c4 = _mm512_fmadd_ps(a4, b, c4);
        __m512 a5 = _mm512_set1_ps(a_ptr[5]); c5 = _mm512_fmadd_ps(a5, b, c5);
        a_ptr += MR;
    }

    _mm512_storeu_ps(C + 0*ldc, c0);
    _mm512_storeu_ps(C + 1*ldc, c1);
    _mm512_storeu_ps(C + 2*ldc, c2);
    _mm512_storeu_ps(C + 3*ldc, c3);
    _mm512_storeu_ps(C + 4*ldc, c4);
    _mm512_storeu_ps(C + 5*ldc, c5);
}
#endif
}

// ─── Simple reference (scalar) GEMM — used by benchmark for comparison ───────
void scalar_sgemm(int M, int N, int K,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float*       C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = acc;
        }
    }
}
