# DESIGN.md

## 1. The GEMM Performance Problem

Dense matrix multiplication is the core primitive in many ML workloads. The naïve triple-loop algorithm:

```cpp
for (int i = 0; i < M; ++i)
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      C[i][j] += A[i][k] * B[k][j];
```

is simple, but it performs poorly on modern CPUs because it does not reuse data effectively. It streams the same input rows and columns repeatedly from memory, causing high traffic to DRAM and low arithmetic intensity.

For an FP32 GEMM of size `M×K×N`, the work is `2*M*N*K` FLOPs and the minimum data movement is approximately `4*(M*K + K*N + M*N)` bytes. The arithmetic intensity (AI) is therefore:

```text
AI = 2*M*N*K / (4*(M*K + K*N + M*N)) FLOP/byte
```

For square matrices with `M=N=K=S`, the AI simplifies to:

```text
AI = S / 2 FLOP/byte
```

That means a 256×256 GEMM has `AI ≈ 128` FLOP/byte, which is already in the compute-bound regime on most CPUs. For smaller sizes the bottleneck is memory; for larger sizes the bottleneck is compute.

A production-quality GEMM kernel therefore needs to maximize reuse through blocking, packing, and careful register scheduling.

## 2. The Goto Algorithm

The Goto algorithm (Goto & van de Geijn, 2008) is a high-performance GEMM strategy that partitions the computation into three levels of blocking:

- `NC` for the outermost loop, sized to fit L3 cache
- `KC` for the shared inner block, sized to fit L2 cache
- `MC` for the panel of A, sized to fit L1 cache

This repository uses the following static tile sizes:

- `MR = 8`
- `NR = 8`
- `KC = 256`
- `MC = 128`
- `NC = 2048`

Packing transforms submatrices of `A` and `B` into contiguous buffers to improve streaming efficiency and reduce TLB pressure. Packing also enables a clean inner kernel that can assume contiguous data and reuse vector registers effectively.

The main idea is that each block of `A` and `B` is loaded once from L2/L3 and reused many times by the register-blocked inner kernel.

## 3. AVX2 Register Blocking

AVX2 provides 256-bit vector registers, which can hold eight `float` values. The register blocking strategy used here is an 8×8 micro-kernel:

- `MR = 8` rows of `A` per register block
- `NR = 8` columns of `B` per register block

This choice is driven by the register file on AVX2 CPUs: 16 YMM registers are available, and the kernel needs both A/B inputs plus accumulators. A safe scheme reserves 2 registers for pointer and constant handling, leaving about 14 registers for data. An 8×8 block strikes a good balance between register reuse and instruction throughput.

FMA latency on Haswell/Zen is around 4 cycles. To hide this latency, the kernel maintains multiple independent accumulator registers and interleaves loads with FMA operations.

## 4. Activation Function Numerics

### GeLU

GeLU is defined as:

```text
GeLU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

This implementation uses the fast approximation:

```text
GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The code is vectorized in AVX2 and evaluates the polynomial in Horner form to minimize instructions. The implementation targets numeric stability by clamping values and using `erff` in the scalar reference path.

### SiLU

SiLU is defined as:

```text
SiLU(x) = x / (1 + exp(-x))
```

Efficient evaluation uses a numerically stable approximation for `exp(-x)` in the AVX2 path, combined with vectorized multiplication and blend operations.

### Softmax

Softmax is computed row-wise with the standard stability trick:

```text
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

Subtracting the row maximum avoids overflow and keeps the exponent range bounded.

## 5. Performance Model

The theoretical peak FP32 throughput for an AVX2 core is:

```text
peak = frequency_ghz * 16 FLOP/cycle
```

because AVX2 FMA can perform 8 multiply-add operations per cycle, which counts as 16 FP32 operations.

For a 3.6 GHz core, the peak is:

```text
3.6 * 16 = 57.6 GFLOPS
```

Measured efficiency is reported as:

```text
utilization = measured_GFLOPS / peak_GFLOPS
```

This repository's current implementation is designed to achieve a significant fraction of peak for moderate block sizes, with the remaining gap attributable to missing architecture-specific tuning and packing optimizations.

## 6. What Is Missing vs Production

This project intentionally omits production features that are important for a full BLAS-quality library:

- Hardware-specific prefetch hints (`PREFETCHT0`, `PREFETCHNTA`)
- NUMA-aware allocation and thread placement
- Runtime cache-size probing and adaptive tile selection
- Architecture-specific tile sizes for Zen, Skylake, and Ice Lake
- AVX-512 micro-kernels and AMX/DPAS paths
- Int8/BF16 inference kernels and quantized support

These omissions explain the remaining 20–30% performance gap versus tuned libraries such as OpenBLAS or MKL.

## Source File References

- `src/kernels/gemm/avx2_gemm_packed.cpp` — packed GEMM, see §2 and §3
- `src/kernels/activations/activations.hpp` — activation APIs, see §4
- `src/bindings/pybind_entry.cpp` — Python API surface, see §10 in feedback
