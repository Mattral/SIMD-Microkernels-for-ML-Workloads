# GEMM Algorithm

This repository implements a blocked FP32 general matrix multiply (GEMM) kernel.
The implementation is intentionally straightforward while still using SIMD intrinsics and cache-aware tiling.

## Blocking strategy

The GEMM driver partitions the input matrices into tiles to improve locality:

* `M` — number of output rows
* `N` — number of output columns
* `K` — shared inner dimension

A typical block loop structure is:

```
for (m = 0; m < M; m += MB)
  for (k = 0; k < K; k += KB)
    for (n = 0; n < N; n += NB)
      compute_block(A[m:m+MB, k:k+KB], B[k:k+KB, n:n+NB], C[m:m+MB, n:n+NB]);
```

The inner compute block is sized to match the AVX2 vector width and the expected register footprint.

## Micro-kernel shape

The implementation uses an 8×8 micro-kernel shape for FP32 data.
This fits well with AVX2 because each YMM register holds 8 floats.

The inner loop performs:

* `A` block loads broadcast across registers
* `B` block loads contiguous vectors
* FMA updates accumulate into output registers

## Goto-style considerations

A full packed-GEMM design would use `A` and/or `B` packing to optimize memory access and reduce TLB pressure.
This repository takes a simpler path:

* Avoid explicit packed buffers for clarity
* Use tiled loops to keep working sets in L1/L2
* Rely on fixed block sizes rather than dynamic packing

This is a pragmatic compromise for a review-oriented codebase: the kernel remains readable while still showing the benefit of SIMD and tiling.

## Key tradeoffs

* Pros:
  * Clear control flow and simple inner kernel
  * Low implementation overhead
  * Good cache reuse for moderate matrix sizes
* Cons:
  * No dedicated pack step, so memory accesses are less optimal than high-performance BLAS
  * Fixed tile sizes limit cross-architecture tuning
  * Not optimized for small or extremely large matrices in the same code path
