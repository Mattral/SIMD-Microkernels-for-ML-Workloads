#pragma once

#include <cstddef>

namespace simd_ml {
namespace gemm {

// ─── Tile/blocking constants ──────────────────────────────────────────────────
// These constants are derived from typical Intel client CPU cache hierarchy.
// See docs/DESIGN.md §2 for the full derivation.
//
//   MR=8, NR=8  — micro-kernel register block: 8 YMM accumulators (one per row),
//                 1 YMM for B panel → 9/16 YMM used, leaves headroom for prefetch
//   KC=256      — K-panel: Ac(MC×KC) + Bc(KC×NC) panel pair fits in L2
//   MC=128      — M-panel: Ac stays L2-resident while B panel streams from L3
//   NC=2048     — N-panel: large to amortise B packing cost across many M-tiles
static constexpr int MR = 8;
static constexpr int NR = 8;
static constexpr int KC = 256;
static constexpr int MC = 128;
static constexpr int NC = 2048;

// Reference: Goto, K. & van de Geijn, R. (2008). Anatomy of High-Performance
// Matrix Multiplication. ACM Transactions on Mathematical Software, 34(3).

// ─── Panel packing ────────────────────────────────────────────────────────────

/** pack_b_panel — Reorder B[k × n] into NR-column-major panel layout. */
void pack_b_panel(const float* B, int ldb, int k, int n, float* B_packed);

/** pack_a_panel — Reorder A[m × k] into row-major micro-panel layout. */
void pack_a_panel(const float* A, int lda, int m, int k, float* A_packed);

// ─── AVX2 8×8 micro-kernel ────────────────────────────────────────────────────

/**
 * inner_kernel_8x8 — Compute C[0..7][0..7] += alpha * A_packed × B_packed
 *
 * A_packed: row-major, shape [MR × k_rem]   (from pack_a_panel)
 * B_packed: NR-column-major, shape [k_rem × NR]  (from pack_b_panel)
 * C:        row-major output with leading dimension ldc
 */
void inner_kernel_8x8(const float* A_packed,
                       const float* B_packed,
                       float*       C, int ldc,
                       int          k_rem,
                       float        alpha) noexcept;

// ─── Main GEMM entry point ────────────────────────────────────────────────────

/**
 * sgemm_packed — Compute C = alpha*A*B + beta*C  (float32, row-major)
 *
 * Implements the Goto/BLIS 5-loop structure with panel packing.
 * This is the kernel used by the Python dispatch layer (kernel_registry.hpp).
 */
void sgemm_packed(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

// ─── OpenMP thread control ────────────────────────────────────────────────────

/**
 * set_num_threads / get_num_threads — control the OpenMP thread count used
 * by the outermost jc-parallel loop in sgemm_packed.
 * No-ops when the library is built without -DSIMD_ML_OPENMP.
 */
void set_num_threads(int n) noexcept;
int  get_num_threads()      noexcept;

}  // namespace gemm
}  // namespace simd_ml
