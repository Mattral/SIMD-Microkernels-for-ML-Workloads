# AVX2 Register File Usage

The x86-64 ABI provides 16 YMM registers (YMM0–YMM15). Each YMM holds 8 single-precision floats (256 bits). The micro-kernel is designed to keep as many of these in the accumulator as possible to minimize register spills.

---

## 8×8 Micro-kernel (avx2_gemm_packed.cpp)

```
Register allocation at any point in the K-loop:
  acc0 .. acc7    8 YMM    output accumulators (one per row of the 8×8 tile)
  c0   .. c7      8 YMM    existing C values  (loaded before K-loop, written after)
  b               1 YMM    B panel column (8 floats, loaded once per k-step)
  a_broadcast     implicit  set1_ps scalar broadcast into b-register slot

  Total used:     9 / 16 YMM registers
  Spill pressure: none — 7 spare for address computation and ABI requirements
```

The 8×8 layout leaves 7 YMMs free, which provides:
- One register for the next B panel prefetch target address
- Room for the compiler to hold loop counters and array pointers
- No register spill to the stack in the hot K-loop

---

## 6×16 Micro-kernel (avx_matmul.cpp)

```
Register allocation:
  c0_lo, c0_hi   2 YMM    row 0 accumulators (16 floats = 2 YMMs wide)
  c1_lo, c1_hi   2 YMM    row 1
  ...
  c5_lo, c5_hi   2 YMM    row 5
                 ─────────
  12 YMM total   for accumulators

  b_lo, b_hi     2 YMM    B panel (16 floats = 2 YMMs wide)
  a_bcast        1 YMM    scalar broadcast

  Total used:    15 / 16 YMM — one spare for loop control
  Spill pressure: minimal — one register headroom on Haswell/Skylake (16 physical YMMs)
```

The 6×16 layout maximises accumulator utilization (15/16 YMMs vs 9/16 for 8×8),
achieving more FP operations per k-step (12 FMAs vs 8). The trade-off is
tighter register pressure: on CPUs with physical register files smaller than
the architectural limit, this can cause spills.

**Empirical guidance**: prefer 8×8 for general use; 6×16 is better when N is
much larger than M (wide output rows) where the wider NR=16 tile amortises
the overhead of loading two YMM vectors per B column across more output elements.

---

## FMA Throughput Analysis

On Skylake-class CPUs (and most AMD Zen 3+ cores):

```
FMA execution units:  2 ports (port 0, port 5)
FMA throughput:       0.5 cycles per _mm256_fmadd_ps
FMA latency:          4 cycles (3 cycles on Zen 3)

Peak FP throughput:   2 ports × 8 floats × 2 ops = 32 FP-ops/cycle
```

To saturate the FMA units, the kernel needs enough independent instructions
in-flight to cover the 4-cycle latency: at 0.5 cy/issue, that requires
4 / 0.5 = 8 independent chains. The 8×8 kernel has 8 accumulator chains
(acc0..acc7) — exactly the minimum needed to hide FMA latency.

The 4× K-loop unrolling in both kernels ensures the out-of-order scheduler sees
32 FMAs at once, providing maximum ILP scheduling flexibility.

---

## AVX-512 Status

This project currently does **not** use AVX-512 in production. A stub
`gemm_micro_6x32_avx512` exists in `avx_matmul.cpp` but is intentionally
disabled — it accumulates into a single `__m512` per row (16 floats), when
a correct 6×32 tile requires two `__m512` accumulators per row. The full
dual-accumulator AVX-512 kernel is listed in ROADMAP.md as v0.8 work.

On AVX-512 hardware, all 256-bit operations in this library execute at full
throughput because the 512-bit EVEX encoding of 256-bit ops runs on the
same ports as native AVX2.
