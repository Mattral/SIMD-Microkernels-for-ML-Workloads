# Cache Hierarchy

Effective GEMM performance depends on matching tile shapes to the CPU cache hierarchy.
This repository uses a blocking strategy that keeps the active working set small enough for L1/L2 reuse.

## Cache levels

* `L1` — smallest and fastest cache, ideal for the inner micro-kernel working set
* `L2` — larger scratch space for next-level tiles
* `L3` — shared cache for the full problem, not targeted explicitly in this implementation

## Tiling rationale

The blocked loop structure ensures that the innermost compute kernel operates on data already loaded into the lowest-level cache.
This reduces bandwidth demands on main memory and improves sustained throughput.

### Working set components

For a blocked GEMM, the critical data in the inner loop is:

* A tile of `A`
* A tile of `B`
* The output tile of `C`

By keeping `A` and `B` tiles small enough and reusing them across multiple output columns or rows, the kernel improves cache locality.

## Alignment

Aligned allocations are used to avoid split-cache-line loads and stores.
A 64-byte alignment is chosen to match typical cache-line sizes and ensure that AVX loads and stores can operate cleanly.

## Practical impact

This design is well-suited for moderate-sized matrices where the blocked tile fits in L1/L2.
It is not a full L3- or NUMA-aware design, but it does demonstrate the core principle: minimize main memory traffic by reusing data in cache.
