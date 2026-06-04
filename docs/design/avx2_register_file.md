# AVX2 Register File

The AVX2 micro-kernel is designed around the 16 YMM registers available in x86-64.
Each register holds eight 32-bit floats.

## Register usage pattern

The inner loop targets a small working set that can be kept in registers and L1 cache.
A typical pattern is:

* Load a block of `B` into vector registers
* Broadcast elements of `A` across vector lanes
* Use FMA to accumulate products into output registers

AVX2 allows eight float elements per register, so the design naturally maps to 8×8 inner tiles.

## FMA and latency

FMA instructions on AVX2 provide a fused multiply-add in a single instruction.
This is critical for GEMM because the inner loop is dominated by `C += A * B` updates.

Important points:

* The kernel exposes high instruction-level parallelism by maintaining multiple accumulators.
* A well-formed inner loop overlaps loads and arithmetic to keep execution units busy.

## Why AVX2

AVX2 provides the following benefits for this project:

* 256-bit vectors for 8 FP32 values
* `_mm256_load_ps` / `_mm256_store_ps` for contiguous memory
* `_mm256_fmadd_ps` for combined multiply-add
* Sufficient register space for blocked gather-style compute

The current implementation is not AVX-512 optimized, but the register-file rationale is the same: choose a tile size that matches the SIMD width and keep the arithmetic pipeline full.
