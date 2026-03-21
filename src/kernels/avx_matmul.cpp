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
    for (int k = 0; k < kc; ++k) {
        for (int n = 0; n < nc; ++n) {
            B_packed[k * nc + n] = B[k * ldb + n];
        }
        // Pad remainder columns with zeros for register-width alignment
        for (int n = nc; n < NR; ++n) {
            B_packed[k * NR + n] = 0.0f;
        }
    }
}

/**
 * pack_A — reorder A[m0..m0+mr, k0..k0+kc] into column-major micro-panel.
 * Gives the inner K-loop sequential access for each row broadcast.
 */
static void pack_A(const float* A, int lda,
                   int mr, int kc,
                   float* __restrict__ A_packed) {
    for (int k = 0; k < kc; ++k) {
        for (int m = 0; m < mr; ++m) {
            A_packed[k * MR + m] = A[m * lda + k];
        }
        for (int m = mr; m < MR; ++m) {
            A_packed[k * MR + m] = 0.0f;
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

    // Allocate packing buffers (L2-resident)
    auto A_pack = make_aligned_array<float>(MC * KC);
    auto B_pack = make_aligned_array<float>(KC * NC);

    // 3-level tiled loop: N-tile → K-tile → M-tile → micro-kernel
    for (int j = 0; j < N; j += NC) {
        int nc = std::min(NC, N - j);

        for (int k = 0; k < K; k += KC) {
            int kc = std::min(KC, K - k);

            // Pack B panel into L2-friendly layout
            pack_B(B + k * ldb + j, ldb, kc, nc, B_pack.get());

            for (int i = 0; i < M; i += MC) {
                int mc = std::min(MC, M - i);

                // Pack A panel
                pack_A(A + i * lda + k, lda, mc, kc, A_pack.get());

                // Micro-kernel loop over the MC × NC tile.
                // pack_A stores in column-major micro-panel order:
                //   A_pack[k * MR + m]  for k in [0,kc), m in [0,MR)
                // So the ii-th MR-row block starts at byte offset ii * kc * MR floats —
                // but since we process one MR block at a time, the pointer is:
                //   A_pack + (ii / MR) * (kc * MR)
                for (int ii = 0; ii < mc; ii += MR) {
                    int mr = std::min(MR, mc - ii);
                    // Pointer into the packed A panel for this MR-block:
                    // Each prior MR block occupies kc * MR floats.
                    const float* A_micro = A_pack.get() + (ii / MR) * (kc * MR);
                    for (int jj = 0; jj < nc; jj += NR) {
#ifdef __AVX2__
                        gemm_micro_6x16_avx2(
                            kc,
                            A_micro,
                            B_pack.get() + jj,
                            C + (i + ii) * ldc + (j + jj),
                            ldc
                        );
#endif
                    }
                }
            }
        }
    }

    // Apply alpha scalar
    if (alpha != 1.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= alpha;
    }
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
