# GEMM Algorithm: Goto/BLIS 5-Loop Structure

IntrinsicML implements the Goto (2008) / BLIS-style packed GEMM in two complementary kernels. This document describes the shared algorithmic structure.

---

## The 5-Loop + 2 Packing Structure

The central insight from Goto & van de Geijn (2008) is that a high-performance GEMM must decomposes into **five nested loops** plus **two packing routines** that together ensure every level of the cache hierarchy is used efficiently:

```
for jc in 0..N step NC:          # Loop 1: N in NC tiles  (B panel → L3)
  for pc in 0..K step KC:        # Loop 2: K in KC tiles  (B_packed → L2)
    pack_b_panel(B[pc:, jc:])
    for ic in 0..M step MC:      # Loop 3: M in MC tiles  (A_packed → L2)
      pack_a_panel(A[ic:, pc:])
      for jr in 0..nc step NR:   # Loop 4: nc in NR steps (register tile)
        for ir in 0..mc step MR: # Loop 5: mc in MR steps (register tile)
          inner_kernel()          # ← hot loop: pure FMA
```

The two packing routines (`pack_b_panel`, `pack_a_panel`) convert the strided matrix access pattern into sequential streaming access, enabling the prefetcher and eliminating TLB misses in the inner kernel.

---

## Why Packing Matters

Without packing, the inner kernel accesses B with a stride of `ldb` floats between columns. For a 512×512 matrix (8 MB), this causes:
- Every B column access misses L1 (stride = 2 KB, L1 = 32–48 KB)
- TLB pressure from widely-scattered pages
- Observed throughput: ~5% of theoretical FMA peak

After packing B into a contiguous `kc × nr` panel:
- The inner kernel's B reads are sequential (unit stride)
- The hardware prefetcher can fully hide L2→L1 latency
- Observed throughput: 40–60% of theoretical FMA peak (single-threaded)

---

## Tile Size Derivation

| Constant | 8×8 kernel | 6×16 kernel | Derivation |
|---|---|---|---|
| MR | 8 | 6 | Register accumulators: MR × (NR/8) YMMs must fit in 16 YMM file |
| NR | 8 | 16 | 8 floats/YMM; 6×16 uses 12 accumulators, wider ILP |
| KC | 256 | 256 | Ac+Bc panels fit in L2 (256 KB): MC×KC×4 + KC×NC×4 ≤ L2 |
| MC | 128 | 64 | Ac panel (MC×KC×4 = 128 KB) fits in L2 alongside Bc |
| NC | 2048 | 64 | Large NC amortises B packing cost across many M-tiles |

The NC=2048 choice in `avx2_gemm_packed.cpp` (vs NC=64 in `avx_matmul.cpp`) reflects a trade-off: larger NC means fewer B packing passes but larger L3 working set. For the Python-facing kernel, L3-resident B panels are acceptable since the outer loop is parallelised with OpenMP.

---

## Panel Packing Layout

### B panel (`pack_b_panel`)

Input:  B[k×ldb] — strided row-major access  
Output: B_packed[(nc/NR) × kc × NR] — block-sequential

```
B_packed[block * kc * NR + p * NR + j] = B[p][block*NR + j]
```

Each `kc × NR` block is loaded sequentially by the inner kernel for one `jr` tile iteration. The NR-wide stride within each block enables a single `_mm256_loadu_ps` to load all 8 B values for one k-step.

### A panel (`pack_a_panel`)

Input:  A[m×lda] — strided row-major access  
Output: A_packed[mc × kc] — contiguous row-major

```
A_packed[i * kc + p] = A[i][p]
```

This layout allows scalar broadcast loads `_mm256_set1_ps(a_row[p])` for each row of the micro-kernel with sequential `p` access.

---

## Inner Kernel: 8×8 AVX2 (`avx2_gemm_packed.cpp`)

```
Registers:
  acc0..acc7   8 YMM   (8 row accumulators)
  c0..c7       8 YMM   (existing C values)
  b            1 YMM   (B panel column)

For each p in 0..kc-1:
  b = loadu(B_block + p*NR)            # 8 floats from B panel
  acc0 = fmadd(set1(a0[p]), b, acc0)   # row 0: scalar-broadcast A × vector B
  acc1 = fmadd(set1(a1[p]), b, acc1)   # row 1
  ...
  acc7 = fmadd(set1(a7[p]), b, acc7)   # row 7

C[0..7][0..7] += alpha * acc[0..7]     # write-back
```

8 FMAs per k-step, each computing one row of the 8×8 output tile. At 0.5 cycle/FMA throughput on two FMA ports, the theoretical ceiling is 16 FP32 ops/cycle = 56 GFLOPS at 3.5 GHz.

The 4× K-loop unrolling reduces branch overhead and exposes 32 FMAs to the out-of-order scheduler simultaneously, better hiding the 12-cycle FMA dependency chain.

---

## Inner Kernel: 6×16 AVX2 (`avx_matmul.cpp`)

An alternative tile shape that trades accumulator count for wider NR:

```
Registers:
  c0_lo, c0_hi ... c5_lo, c5_hi   12 YMM  (6 rows × 2 vectors each)
  b_lo, b_hi                        2 YMM  (16 floats from B panel)
  a_bcast                           1 YMM  (scalar broadcast)

For each k:
  b_lo = load(B + k*16),  b_hi = load(B + k*16 + 8)
  a_bcast = set1(A[k*6 + row])
  c_row_lo = fmadd(a_bcast, b_lo, c_row_lo)   # 8 cols
  c_row_hi = fmadd(a_bcast, b_hi, c_row_hi)   # 8 more cols
```

12 FMAs per k-step (6 rows × 2 vectors). With two FMA ports this processes 16 columns simultaneously — wider than the 8×8 kernel but with higher register pressure (15/16 YMM used).

The `alpha` scalar is folded into `pack_A` at packing time, eliminating a per-k multiply in the inner loop.
