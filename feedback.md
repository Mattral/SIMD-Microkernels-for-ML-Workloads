---

# ENGINEERING DIRECTIVE: SIMD-Microkernels Remediation & Roadmap

**From:** Principal Review  
**Priority:** P0 (blocking correctness) → P1 (core kernel engineering) → P2 (measurement & documentation)  
**Context:** A deep audit found that this repo's three most important requirements — a true register-blocked GEMM microkernel with matrix packing, calibrated cycle-accurate GFLOPS measurements, and a roofline-grounded README — are entirely absent. The build system and project structure are solid and require no changes. All remediation work is in the kernel code, the benchmark harness, and the documentation. Execute in strict P0 → P1 → P2 order.

---

## IMMEDIATE FIXES (Do Before Any New Code)

### Fix 1: Remove `-ffast-math` from the Precision Test Target

**File:** `CMakeLists.txt`

The current build applies `-ffast-math` globally to all targets including the benchmark. This means your precision tests are comparing a non-IEEE SIMD approximation against a non-IEEE scalar reference. The reference is corrupted by the same flag. You cannot measure approximation error against a distorted ground truth.

**Change:** Split compiler flags into two sets:

```cmake
# Performance flags — for kernels and bench only
set(PERF_FLAGS "-O3" "-march=native" "-mfma" "-ffast-math" "-DNDEBUG")

# Precision flags — for test targets only; NO -ffast-math
set(PRECISION_FLAGS "-O2" "-march=native" "-mfma" "-DNDEBUG")
```

Apply `PERF_FLAGS` to `simd_kernels_lib`, `simd_kernels` (pybind extension), and `bench`.  
Apply `PRECISION_FLAGS` to every target under `tests/`.

The reference scalar implementation used in precision tests must be compiled with `-O2` and no `-ffast-math` so it computes IEEE 754-conforming results that serve as genuine ground truth.

**Acceptance:** `tests/test_precision.py` must compare SIMD output against a `numpy.float64` reference (not `float32`) and must assert `max_abs_error < 1e-5` for GeLU across the range `[-5.0, 5.0]` in steps of 0.001.

---

### Fix 2: Validate Buffer Contiguity in Python Bindings

**File:** `src/bindings/pybind_entry.cpp`

NumPy arrays that are slices or transposes are not C-contiguous. Passing a non-contiguous buffer's raw pointer to a SIMD kernel that assumes row-major stride will silently produce wrong results — no crash, no error, wrong numbers.

Add contiguity and type checks at every binding entry point:

```cpp
// At the top of every bound function that accepts a numpy array:
if (!A.dtype().is(py::dtype::of<float>()))
    throw std::runtime_error("Expected float32 array");
if (!A.attr("flags").attr("c_contiguous").cast<bool>())
    throw std::runtime_error(
        "Input array must be C-contiguous. "
        "Call numpy.ascontiguousarray(arr) before passing.");
// Also check shape dimensions are compatible with kernel tile size
```

Also add an alignment check: if the user passes a pre-existing NumPy array whose data pointer is not 32-byte aligned (required for `_mm256_load_ps`), fall back to `_mm256_loadu_ps` (unaligned load) rather than crashing with a SIGBUS. Add a `bool is_aligned = (reinterpret_cast<uintptr_t>(ptr) % 32 == 0)` check and branch.

**Acceptance:** Add `tests/test_bindings_edge_cases.py` that deliberately passes a non-contiguous slice and verifies a `RuntimeError` is raised with the descriptive message.

---

## P0 — BENCHMARK HARNESS: Make RDTSC Actually Cycle-Accurate

**File:** `src/main_bench.cpp`

The current RDTSC harness reads the timestamp counter but does not calibrate it to actual core frequency. On modern CPUs with Turbo Boost, the TSC frequency (nominally the rated base clock) diverges from the actual execution frequency. You cannot compute GFLOPS from raw TSC ticks without knowing ticks-per-second.

### P0-1: Calibrate TSC Frequency

Add a one-time calibration function at program start that measures TSC ticks over a known wall-clock interval using `clock_gettime(CLOCK_MONOTONIC)`:

```cpp
static uint64_t calibrate_tsc_hz() {
    // Warm up to reach turbo state
    volatile double sink = 0.0;
    for (int i = 0; i < 1000000; ++i) sink += i * 0.001;

    struct timespec t0, t1;
    uint64_t tsc0 = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Sleep for a known interval — 100ms is sufficient
    struct timespec req = {0, 100'000'000L};
    nanosleep(&req, nullptr);

    uint64_t tsc1 = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    uint64_t ns_elapsed = (t1.tv_sec - t0.tv_sec) * 1'000'000'000ULL
                        + (t1.tv_nsec - t0.tv_nsec);
    uint64_t tsc_elapsed = tsc1 - tsc0;

    return (uint64_t)((double)tsc_elapsed / (double)ns_elapsed * 1e9);
}
```

Store this as `const uint64_t TSC_HZ = calibrate_tsc_hz()` before any benchmark runs.

### P0-2: Use CPUID Serialization Around RDTSC

Raw `__rdtsc()` can be reordered by the CPU's out-of-order execution engine, making cycle counts unreliable for short sequences. Replace bare `__rdtsc()` with serialized reads:

```cpp
static inline uint64_t rdtsc_start() {
    unsigned aux;
    // CPUID serializes: prevents instructions before this point
    // from being measured as part of the timed region
    _mm_lfence();  // Lighter than CPUID; sufficient for AMD+Intel
    return __rdtsc();
}

static inline uint64_t rdtsc_end() {
    uint64_t tsc = __rdtscp(&(unsigned){0}); // RDTSCP: waits for prior stores
    _mm_lfence();  // Prevent subsequent instructions from leaking back
    return tsc;
}
```

Use `rdtsc_start()` / `rdtsc_end()` for all measurements. Never use bare `__rdtsc()` in timing code.

### P0-3: Pin the Benchmark Thread to One Core

Without CPU affinity, the OS scheduler may migrate the benchmark thread between cores mid-run, causing a TSC discontinuity. Add affinity pinning at the start of `main()`:

```cpp
#ifdef __linux__
#include <sched.h>
static void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        std::cerr << "Warning: failed to pin to core " << core_id << "\n";
}
#endif
// In main(): pin_to_core(0);
```

### P0-4: Report Actual Measured GFLOPS

Every benchmark run must compute and print real GFLOPS:

```cpp
// For an M×K×N GEMM:
const double flops = 2.0 * M * N * K;  // 1 multiply + 1 add per element
const double elapsed_seconds = (double)(tsc_end - tsc_start) / (double)TSC_HZ;
const double gflops = (flops / elapsed_seconds) / 1e9;

printf("GEMM %4zu×%4zu×%4zu | cycles: %8zu | time: %.3f ms | %.2f GFLOPS\n",
       M, N, K, (size_t)(tsc_end - tsc_start),
       elapsed_seconds * 1000.0, gflops);
```

Report: raw cycle count, elapsed milliseconds, achieved GFLOPS, and theoretical peak GFLOPS (hardcoded for your build machine: `peak = cores × freq_GHz × 16` for AVX2 FP32, or `× 32` for AVX-512 FP32). Compute and print utilization percentage: `(achieved / peak) * 100`.

**Acceptance:** Running `./bench` must output a table with real numbers, no "~" ranges, no "Indicative" labels. The output must include utilization percentage so the reader immediately knows how far below theoretical peak the kernel runs.

---

## P1 — CORE KERNEL: Implement a Real GEMM Microkernel

This is the most important engineering work in the entire repo. The current tiled triple-loop is not a microkernel. It is a cache-blocked loop. The difference is packing and explicit register accumulation.

### P1-1: Implement Matrix B Panel Packing

**File:** `src/kernels/avx_matmul.cpp` (new function `pack_B_panel`)

The reason production GEMM kernels achieve near-peak FLOP/s is that the B matrix is repacked into a layout where every cache line contains consecutive data in the order the inner kernel accesses it. Without packing, accessing B's columns causes stride-N cache misses.

```cpp
// Pack a (kc × nc) panel of B into contiguous column-major layout
// so that the inner kernel streams B sequentially.
// kc: block size along K dimension (must fit with A panel in L1 together)
// nr: inner kernel width (8 for AVX2, 16 for AVX-512)
void pack_B_panel(
    const float* B,   // original B, row-major, stride N
    float*       Bp,  // packed output buffer, must be 64-byte aligned
    int K, int N,     // original dimensions
    int ib, int kc, int jb, int nc, int nr
) {
    // For each k in [ib, ib+kc), for each j in [jb, jb+nc) in nr-wide strips:
    // write B[k][j:j+nr] contiguously into Bp
    // Result: Bp is laid out as [kc][nc/nr][nr] = kc * nc floats
    // Inner kernel reads Bp as a stream with unit stride
    for (int k = 0; k < kc && (ib + k) < K; ++k) {
        for (int j = 0; j < nc; j += nr) {
            int actual_nr = std::min(nr, nc - j);
            const float* src = B + (ib + k) * N + (jb + j);
            float* dst = Bp + k * nc + j;
            std::memcpy(dst, src, actual_nr * sizeof(float));
            // Zero-pad to nr boundary for safe SIMD loads
            if (actual_nr < nr)
                std::memset(dst + actual_nr, 0, (nr - actual_nr) * sizeof(float));
        }
    }
}
```

Implement the equivalent `pack_A_panel` that repacks an `(mc × kc)` panel of A into row-major layout matching the inner kernel's access order.

### P1-2: Implement the 8×8 AVX2 Inner Kernel with Explicit Accumulators

**File:** `src/kernels/avx_matmul.cpp` (new function `gemm_inner_8x8`)

This is the critical missing piece. The inner kernel must keep 8 `__m256` accumulator registers live across the K loop, loading A as scalars (broadcast) and B as full 8-wide vectors:

```cpp
// Compute C[mr×nr] += A[mr×kc] * B[kc×nr]
// mr=6, nr=8 is optimal for AVX2 (uses 6×1 + 6 = 18 of 16 YMM registers;
// use mr=4, nr=8 for safety to leave registers for A loads)
// Here we implement mr=4 to be safe with register pressure
static void gemm_inner_4x8(
    int kc,
    const float* __restrict__ A_panel,  // packed, [kc][4], col-major
    const float* __restrict__ B_panel,  // packed, [kc][8], row-major
    float* __restrict__ C,              // output slice [4][N], unpackaged
    int N                               // stride of C
) {
    // 4 rows × 1 AVX register = 4 __m256 accumulators
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        __m256 b = _mm256_load_ps(B_panel + k * 8);  // 8 B values, aligned

        // Broadcast each A element and fuse multiply-add
        __m256 a0 = _mm256_broadcast_ss(A_panel + k * 4 + 0);
        __m256 a1 = _mm256_broadcast_ss(A_panel + k * 4 + 1);
        __m256 a2 = _mm256_broadcast_ss(A_panel + k * 4 + 2);
        __m256 a3 = _mm256_broadcast_ss(A_panel + k * 4 + 3);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }

    // Accumulate into C (C may have existing partial sums from prior panels)
    float buf[4][8] __attribute__((aligned(32)));
    _mm256_store_ps(buf[0], c0);
    _mm256_store_ps(buf[1], c1);
    _mm256_store_ps(buf[2], c2);
    _mm256_store_ps(buf[3], c3);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            C[i * N + j] += buf[i][j];
}
```

Document every variable's purpose and the register mapping in inline comments. A reviewer must be able to trace exactly which YMM register holds which piece of data.

### P1-3: Derive Tile Sizes from Target Cache Capacity

**File:** `src/kernels/avx_matmul.cpp` (top-of-file constants with derivation comments)

Tile sizes must not be arbitrary. They must be derived from your target CPU's cache capacities, documented with the math:

```cpp
// ─── Tile size derivation ────────────────────────────────────────────────────
// Target CPU: Intel Core (Haswell/Skylake microarchitecture, representative)
//   L1 data cache:  32 KB = 32768 bytes, 8-way, 64-byte cache lines
//   L2 cache:      256 KB = 262144 bytes (unified)
//   L3 cache:        6 MB+ (shared, LLC)
//
// BLIS-style 3-level tiling:
//
//   nr = 8   (inner kernel width = 1 AVX2 register = 8 × float32)
//   mr = 4   (inner kernel height = register pressure budget)
//
//   kc: A_panel [mc × kc] + B_panel [kc × nc] must fit in L2
//       kc × nc × 4 bytes ≤ L2/2 = 131072 bytes
//       kc × 256 × 4 = 131072  →  kc = 128
//       (nc = 256 = 32 × nr, fits 32 B-panel strips)
//
//   mc: A_panel [mc × kc] must fit in L1
//       mc × kc × 4 bytes ≤ L1/2 = 16384 bytes
//       mc × 128 × 4 = 16384  →  mc = 32
//
//   nc = 256 (fits B_panel in L2 alongside working set of C)
//
// These values should be tuned per microarchitecture.
// For AMD Zen 3 (L1=32KB, L2=512KB): kc=256, mc=32, nc=512.

static constexpr int MR = 4;    // inner kernel height (register rows)
static constexpr int NR = 8;    // inner kernel width (1 AVX2 register)
static constexpr int MC = 32;   // A-panel rows (fit in L1)
static constexpr int KC = 128;  // shared K block (fit in L2)
static constexpr int NC = 256;  // B-panel columns (fit in L2)
```

**Every tile size constant must have a derivation comment showing the cache arithmetic.** If you change a constant, the comment must be updated to show the new math.

### P1-4: Wire Packing Into the Outer Loop

Replace the current tiled triple-loop outer structure with the BLIS-style 5-loop algorithm:

```
Loop 5 (jc): j = 0..N step NC
  Pack B panel: pack_B_panel(B, Bp, K, N, 0, KC, jc, NC, NR)
  Loop 4 (ic): i = 0..M step MC
    Pack A panel: pack_A_panel(A, Ap, M, K, ic, MC, 0, KC, MR)
    Loop 3 (pc): p = 0..K step KC   ← this is where B-panel packing actually belongs
      Loop 2 (jr): j2 = 0..NC step NR
        Loop 1 (ir): i2 = 0..MC step MR
          gemm_inner_4x8(KC, Ap + ..., Bp + ..., C + ..., N)
```

This is the structure that enables both kernels to hit L1/L2 cache. Without this loop structure, packing is useless.

### P1-5: Add Software Prefetching

Inside `gemm_inner_4x8`, add prefetch hints for the next iteration's B panel data:

```cpp
// Prefetch next iteration's B data into L1 (distance = 1 panel ahead)
if (k + 8 < kc)
    _mm_prefetch((const char*)(B_panel + (k + 8) * 8), _MM_HINT_T0);
```

Document the prefetch distance in cycles and explain the latency you are hiding.

---

## P2 — DOCUMENTATION: Write What the Prompt Actually Required

### P2-1: Rewrite the Performance Table With Real Numbers

**File:** `README.md`

After P0 and P1 are complete and the benchmark has been run, replace the fabricated table with actual output from `./bench` on your build machine. The table must:

- Name the exact CPU (e.g., "Intel Core i7-1165G7, AVX2, 4.7 GHz boost")
- Show naïve triple-loop cycles, `-O3` auto-vectorized cycles, and your SIMD microkernel cycles
- Show achieved GFLOPS and utilization % of theoretical peak
- Show sizes: at minimum 64³, 128³, 256³, 512³, 1024³

The table must be labeled with the actual machine specs, not "a desktop x86 CPU." No ranges. No "~". No "Indicative only."

### P2-2: Add the Cache-Arithmetic Micro-Architecture Section

**File:** `README.md`

Add a section titled "Memory Hierarchy & Tile Size Derivation" that explains exactly how the tile constants were chosen. Copy the derivation from the `avx_matmul.cpp` comments and expand it into prose with a table:

```
| Cache Level | Capacity  | Resident Data                        | Tile Constant |
|-------------|-----------|--------------------------------------|---------------|
| L1 (32 KB)  | 32768 B   | A panel [MC × KC × 4B] = 16 KB      | MC=32, KC=128 |
| L2 (256 KB) | 262144 B  | B panel [KC × NC × 4B] = 131 KB     | NC=256        |
| L3 (6+ MB)  | variable  | Full C matrix, streaming             | —             |
```

### P2-3: Add the Roofline Analysis Section

**File:** `README.md`

Add a section titled "Roofline Analysis" with the following structure:

For each kernel (naïve GEMM, packed GEMM, GeLU), compute:
- **Operational intensity (OI):** FLOPs ÷ bytes of memory traffic
  - Naïve GEMM: OI ≈ N/3 FLOPs/byte for N×N matrices (low, memory-bound)
  - Packed GEMM: OI ≈ N/2 FLOPs/byte after packing reduces traffic
  - GeLU: OI ≈ ~10 FLOPs/byte (elementwise, clearly memory-bound)
- **Roofline ceiling:** `min(peak_flops, peak_bandwidth × OI)`
- **Achieved:** from the benchmark table
- **Gap:** percentage of roofline achieved

This analysis must use the actual measured bandwidth of your build machine (use `stream` benchmark or cite the CPU's published memory bandwidth spec).

### P2-4: Add a `BENCHMARKS.md` File

Create `BENCHMARKS.md` at the repo root. This file must contain:
- The full `./bench` output, pasted verbatim, with the build machine's CPU model, `lscpu` output, and compiler version (`g++ --version` or `clang++ --version`)
- The CMake build flags that produced the benchmark binary (copy from `CMakeLists.txt`)
- A note on whether Turbo Boost was enabled or disabled during measurement (for reproducibility)

No fabricated numbers. Copy-paste from an actual run.

---

## ROADMAP.md — Required Entries

Create `ROADMAP.md` at the repo root. Entries must be honest about current status:

```markdown
# Roadmap

## Legend
✅ Complete + verified   ⚠️ Partial   ❌ Not started

---

## v0.2 — Correctness & Measurement Foundation (current target)
- [❌] -ffast-math removed from precision test targets (Fix 1)
- [❌] Buffer contiguity validation in Python bindings (Fix 2)
- [❌] TSC frequency calibration (P0-1)
- [❌] CPUID/lfence-serialized RDTSC (P0-2)
- [❌] CPU affinity pinning (P0-3)
- [❌] Real GFLOPS output with utilization % (P0-4)

## v0.3 — True GEMM Microkernel
- [❌] Matrix B panel packing (P1-1)
- [❌] Matrix A panel packing (P1-1)
- [❌] 4×8 AVX2 inner kernel with explicit accumulators (P1-2)
- [❌] Cache-derived tile size constants with derivation comments (P1-3)
- [❌] BLIS-style 5-loop outer structure (P1-4)
- [❌] Software prefetching in inner loop (P1-5)

## v0.4 — Verified Performance Documentation
- [❌] Real benchmark table with CPU model and measured GFLOPS (P2-1)
- [❌] Cache-arithmetic derivation in README (P2-2)
- [❌] Roofline analysis section in README (P2-3)
- [❌] BENCHMARKS.md with verbatim bench output (P2-4)

## v0.5 — AVX-512 Specialization
- [❌] 4×16 AVX-512 inner kernel (NR=16, __m512 accumulators)
- [❌] Runtime CPU dispatch: detect AVX-512 at startup, dispatch to right kernel
- [❌] Updated tile sizes for AVX-512 (NR=16 changes NC arithmetic)
- [❌] Benchmark comparison: AVX2 vs AVX-512 on same matrix sizes

## v0.6 — Extended Kernels
- [❌] Multithreaded GEMM via OpenMP across outer jc loop
- [❌] NUMA-aware allocation for dual-socket systems
- [❌] BF16 accumulation kernel (relevant for modern ML inference)
- [❌] Benchmark against OpenBLAS (single-threaded, matched build flags)

## Known Honest Limitations (never remove this section)
- No matrix packing until v0.3: kernel is cache-blocked loop, not a microkernel
- Performance numbers are fabricated ranges until v0.4; do not cite them
- GeLU precision test validity uncertain until Fix 1 is applied
- AVX-512 flag is enabled in CMake but no AVX-512-width kernel code exists
```

---

## RULES FOR THE AGENT

1. **Do not update the performance table in README.md until you have run `./bench` on a real machine and have actual output.** If you cannot run on a GPU/AVX2 machine, add a clearly labeled `[NOT YET MEASURED — CPU UNAVAILABLE]` placeholder. Do not invent numbers.

2. **The `gemm_inner_4x8` function must have a comment for every intrinsic call** explaining which register holds what data. A reader who does not know the function's purpose must be able to deduce it from the comments alone.

3. **Every tile size constant must have its cache arithmetic derivation in the comment.** `constexpr int KC = 128;` with no comment is unacceptable.

4. **Do not add AVX-512 kernel code until the AVX2 kernel is verified correct** against the NumPy reference with `max_abs_error < 1e-4` across sizes 64³, 128³, 256³. An incorrect but wider kernel is worse than a correct narrow one.

5. **The order of work is Fix 1 + Fix 2 → P0 → P1 → P2.** Do not write roofline analysis before you have real benchmark numbers. Do not write a performance table before the calibrated harness exists.

6. **Every CI-gated test must pass on a CPU without AVX2** by falling back to scalar paths. The CMake AVX2 detection block already handles this for the library; ensure the test suite also has a scalar fallback for machines where `COMPILER_HAS_AVX2` is false.

---

