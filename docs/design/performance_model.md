# Performance Model

This section explains how the repository measures and reports performance.

## Arithmetic intensity

For an `M×K×N` GEMM, the total work is `2*M*N*K` floating-point operations.
The data movement for FP32 inputs and outputs is approximately:

```
bytes = 4*(M*K + K*N + M*N)
```

Arithmetic intensity (AI) is:

```
AI = 2*M*N*K / (4*(M*K + K*N + M*N))
```

Higher AI means the kernel is more compute-bound than memory-bound.

## Roofline comparison

The benchmark harness computes a per-core theoretical peak based on AVX2 FMA throughput, then reports measured GFLOPS as a percentage of that peak.

For a single core with AVX2 and FP32:

* 8 floats per vector
* 2 FMA operations per vector lane per cycle
* 16 FP32 operations per cycle per core

If the CPU frequency is `f` GHz, the peak per-core rate is:

```
peak_GFLOPS = 16 * f
```

## Benchmark methodology

The measurement harness uses serialized cycle counters:

* `LFENCE` before `RDTSC`
* `RDTSCP` after the region
* repeated trials with the minimum cycle count reported

This reduces noise from out-of-order execution and improves the chance of capturing a stable lower-bound result.

### JSON benchmark output

The benchmark writes structured results with:

* matrix sizes
* elapsed time
* cycle counts
* GFLOPS
* utilization percentage

This enables easy comparison across configurations and hardware.

## Limitations

The current measurement approach is sufficient for relative comparisons, but it is not a full production-grade benchmark suite.
For more rigorous analysis, combine these results with:

* CPU frequency locking
* thread pinning
* statistical analysis of multiple runs
* hardware performance counters
