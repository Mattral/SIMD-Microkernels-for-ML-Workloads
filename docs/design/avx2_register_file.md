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

Both GEMM kernels now dispatch to AVX-512 micro-kernels at runtime on
AVX-512-capable hardware (see ROADMAP.md §v0.8), each using a design suited
to its specific tile shape:

**`avx2_gemm_packed.cpp` (primary, Python-facing) — 8×16 tile, single accumulator/row:**
```
Register allocation:
  acc0..acc7    8 ZMM   output accumulators — one per row
  c0..c7        8 ZMM   existing C values
  b             1 ZMM   B panel column (16 floats)

  Total used:  17 / 32 ZMM registers — 15 registers of headroom
```
One `__m512` (16 floats) already spans the full NR512=16 tile width, so —
unlike the reference kernel below — there is no dual-accumulator split
needed. This is architecturally the simplest possible AVX-512 upgrade from
the AVX2 8×8 design: same broadcast structure, twice the register width.

**`avx_matmul.cpp` (secondary, reference) — 6×32 tile, dual accumulator/row:**
```
Register allocation:
  c0_lo,c0_hi .. c5_lo,c5_hi   12 ZMM   accumulators: 6 rows × 2 halves
  b_lo, b_hi                    2 ZMM   B panel (32 floats = 2 ZMM)
  a_broadcast                   1 ZMM   scalar broadcast

  Total used:  15 / 32 ZMM registers — 17 registers of headroom
```
A 32-column tile exceeds one ZMM's 16-float width, so each row needs two
accumulators (lo = cols 0–15, hi = cols 16–31). This dual-accumulator
design is exactly what an earlier, disabled version of this kernel got
wrong: it declared a 32-wide tile but only ever wrote the "lo" half,
silently leaving the "hi" half of every output tile untouched. Both the
current kernel and that specific bug class are covered by permanent
regression tests (`tests/test_gemm.cpp`, `tests/test_gemm_packed.cpp`) that
force 100% of the test computation through the full-block path with zero
possible edge-block masking.

AVX-512's much larger physical register file (32 ZMM vs AVX2's 16 YMM)
means neither kernel above faces the register-pressure trade-offs that
shaped the AVX2 8×8 vs 6×16 design choice (see above) — there was
comfortable headroom to choose the simplest correct design for each tile
shape rather than needing to economize on registers.
