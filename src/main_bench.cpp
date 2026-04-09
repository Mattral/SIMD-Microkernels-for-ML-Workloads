/**
 * main_bench.cpp — Cycle-Accurate CPU Benchmarking Harness (RDTSC)
 *
 * ─── Why RDTSC and not std::chrono? ──────────────────────────────────────────
 * std::chrono::high_resolution_clock wraps CLOCK_MONOTONIC or QueryPerformanceCounter,
 * both of which have ~20-100 ns resolution and incur a syscall on some kernels.
 * For microkernels that complete in <1 µs, this adds >10% systematic error.
 *
 * RDTSC (Read Time-Stamp Counter) is a userspace instruction that returns the
 * CPU's internal cycle counter. At 3.5 GHz: 1 cycle ≈ 0.286 ns — suitable
 * for measuring operations as short as a few hundred clock cycles.
 *
 * ─── RDTSC Serialisation ─────────────────────────────────────────────────────
 * RDTSC alone is not serialising — the CPU's out-of-order engine may reorder
 * it relative to adjacent instructions. We bracket it with:
 *   CPUID  — full serialisation barrier (serialises before RDTSC)
 *   LFENCE — load fence (ensures prior loads complete before RDTSC)
 *
 * Best practice (Intel SDM Vol.2B §4.3):
 *   // START: CPUID → RDTSC
 *   // STOP:  RDTSCP → LFENCE  (RDTSCP self-serialises stores, LFENCE finishes)
 *
 * ─── Compilation ─────────────────────────────────────────────────────────────
 *   g++ -O3 -march=native -mfma -o bench src/main_bench.cpp \
 *       src/kernels/avx_matmul.cpp src/kernels/intrinsic_gelu.cpp
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cassert>
#include <ctime>
#include <time.h>
#include <fstream>
#ifdef __linux__
#  include <sched.h>
#endif

#include "kernels/cache_alloc.hpp"
#include "kernels/avx_matmul.hpp"
#include "kernels/intrinsic_gelu.hpp"

// ─── Serialised TSC read primitives ──────────────────────────────────────────

#if defined(__GNUC__) || defined(__clang__)
#  include <x86intrin.h>   // _rdtsc, __rdtscp, _mm_lfence, _mm_cpuid
#  include <cpuid.h>

static inline uint64_t rdtsc_start() {
    _mm_lfence();                    // ensure prior loads complete before reading TSC
    return __rdtsc();
}

static inline uint64_t rdtsc_end() {
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);     // RDTSCP serialises prior instructions
    _mm_lfence();                    // prevent later loads from bypassing the timestamp
    return t;
}

static uint64_t calibrate_tsc_hz() {
    // Warm up the CPU and enter a steady turbo state.
    volatile double sink = 0.0;
    for (int i = 0; i < 1000000; ++i) sink += i * 0.001;

    timespec t0{}, t1{}, req{};
    req.tv_sec = 0;
    req.tv_nsec = 100000000L;  // 100 ms

    uint64_t tsc0 = rdtsc_start();
    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0) {
        perror("clock_gettime");
        return 0;
    }
    nanosleep(&req, nullptr);
    uint64_t tsc1 = rdtsc_end();
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        perror("clock_gettime");
        return 0;
    }

    uint64_t ns_elapsed = static_cast<uint64_t>(t1.tv_sec - t0.tv_sec) * 1000000000ULL
                        + static_cast<uint64_t>(t1.tv_nsec - t0.tv_nsec);
    uint64_t tsc_elapsed = tsc1 - tsc0;
    if (ns_elapsed == 0) return 0;
    return static_cast<uint64_t>((static_cast<double>(tsc_elapsed) / ns_elapsed) * 1e9);
}

#ifdef __linux__
static void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity");
        fprintf(stderr, "Warning: failed to pin to core %d\n", core_id);
    }
}
#endif

static double g_tsc_hz = 0.0;

#elif defined(_MSC_VER)
#  include <intrin.h>
static inline uint64_t tsc_start() {
    int info[4];
    __cpuid(info, 0);
    return __rdtsc();
}
static inline uint64_t tsc_stop() {
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
#else
#  error "Unsupported compiler for RDTSC"
#endif

// ─── Timer utils ─────────────────────────────────────────────────────────────

struct BenchResult {
    double min_cycles;
    double median_cycles;
    double gflops;
};

struct BenchmarkRecord {
    std::string name;
    std::string category;
    int M = 0;
    int N = 0;
    int K = 0;
    int n = 0;
    BenchResult result;
};

static std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    escaped += buf;
                } else {
                    escaped += c;
                }
        }
    }
    return escaped;
}

static void write_json_report(const std::string& path,
                              const std::vector<BenchmarkRecord>& records) {
    std::ofstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open JSON output file: %s\n", path.c_str());
        return;
    }

    file << "{\n";
    file << "  \"benchmarks\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        file << "    {\n";
        file << "      \"name\": \"" << json_escape(record.name) << "\",\n";
        file << "      \"category\": \"" << json_escape(record.category) << "\",\n";
        file << "      \"M\": " << record.M << ",\n";
        file << "      \"N\": " << record.N << ",\n";
        file << "      \"K\": " << record.K << ",\n";
        file << "      \"n\": " << record.n << ",\n";
        file << "      \"min_cycles\": " << record.result.min_cycles << ",\n";
        file << "      \"median_cycles\": " << record.result.median_cycles << ",\n";
        file << "      \"gflops\": " << record.result.gflops << "\n";
        file << "    }";
        if (i + 1 < records.size()) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    file.close();
    printf("Wrote benchmark JSON report to %s\n", path.c_str());
}

// Run `fn` `reps` times, collect cycle counts, return statistics.
template <typename Fn>
static BenchResult measure(const char* name, Fn fn, long long flops,
                            int warmup = 3, int reps = 10,
                            BenchmarkRecord* out_record = nullptr) {
    // Warmup (page tables, branch predictor, iTLB)
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<uint64_t> cycles(reps);
    for (int i = 0; i < reps; ++i) {
        uint64_t t0 = rdtsc_start();
        fn();
        uint64_t t1 = rdtsc_end();
        cycles[i] = t1 - t0;
    }

    std::sort(cycles.begin(), cycles.end());
    double min_cy = static_cast<double>(cycles[0]);
    double med_cy = static_cast<double>(cycles[reps / 2]);

    double elapsed_sec = min_cy / g_tsc_hz;
    double gflops = static_cast<double>(flops) / elapsed_sec * 1e-9;
#if defined(__AVX512F__)
    constexpr double PEAK_FP_PER_CYCLE = 32.0;
#else
    constexpr double PEAK_FP_PER_CYCLE = 16.0;
#endif
    double peak_gflops = (g_tsc_hz / 1e9) * PEAK_FP_PER_CYCLE;
    double utilization = peak_gflops > 0.0 ? (gflops / peak_gflops) * 100.0 : 0.0;

    printf("  %-30s | min: %10.0f cy | time: %8.3f ms | GFLOPS: %6.2f | util: %5.1f%%\n",
           name, min_cy, elapsed_sec * 1000.0, gflops, utilization);

    BenchResult result{min_cy, med_cy, gflops};
    if (out_record) {
        out_record->name = name;
        out_record->result = result;
    }
    return result;
}

// ─── Benchmarks ──────────────────────────────────────────────────────────────

static void bench_gemm(std::vector<BenchmarkRecord>& records,
                         int warmup, int reps) {
    printf("\n=== GEMM Benchmarks (single-precision, C = A*B) ===\n");
    printf("  %-30s | %-22s | %-22s | %s\n",
           "Config", "min cycles", "median cycles", "GFLOPS");
    printf("  %s\n", std::string(85, '-').c_str());

    const int sizes[] = {64, 128, 256, 512, 1024};

    for (int sz : sizes) {
        int M = sz, N = sz, K = sz;
        long long flops = 2LL * M * N * K;  // M*N*K multiplies + M*N*K adds

        auto A = make_aligned_array<float>(M * K);
        auto B = make_aligned_array<float>(K * N);
        auto C_simd   = make_aligned_array<float>(M * N);
        auto C_scalar = make_aligned_array<float>(M * N);

        // Initialise with pseudo-random values
        for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 17) * 0.01f;
        for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(i % 13) * 0.01f;

        char label_simd[64], label_scalar[64];
        snprintf(label_simd,   sizeof(label_simd),   "SIMD   GEMM %4dx%4d", sz, sz);
        snprintf(label_scalar, sizeof(label_scalar),  "Scalar GEMM %4dx%4d", sz, sz);

        // Only run scalar for sizes ≤ 256 (would be very slow for 1024×1024)
        if (sz <= 256) {
            BenchmarkRecord scalar_record{};
            scalar_record.category = "gemm";
            scalar_record.name = label_scalar;
            scalar_record.M = M;
            scalar_record.N = N;
            scalar_record.K = K;
            measure(label_scalar,
                    [&]() {
                        scalar_sgemm(M, N, K,
                                     A.get(), K,
                                     B.get(), N,
                                     C_scalar.get(), N);
                    },
                    flops, warmup, reps, &scalar_record);
            records.push_back(std::move(scalar_record));
        } else {
            printf("  %-30s | (skipped — too slow for scalar)\n", label_scalar);
        }

        BenchmarkRecord simd_record{};
        simd_record.category = "gemm";
        simd_record.name = label_simd;
        simd_record.M = M;
        simd_record.N = N;
        simd_record.K = K;
        auto r_simd = measure(label_simd,
                              [&]() {
                                  simd_sgemm(M, N, K,
                                             1.0f,
                                             A.get(), K,
                                             B.get(), N,
                                             0.0f,
                                             C_simd.get(), N);
                              },
                              flops, warmup, reps, &simd_record);
        records.push_back(std::move(simd_record));
        (void)r_simd;
    }
}

static void bench_gelu(std::vector<BenchmarkRecord>& records,
                         int warmup, int reps) {
    printf("\n=== GeLU Benchmarks (element-wise, FP32) ===\n");
    printf("  %-30s | %-22s | %-22s | %s\n",
           "Config", "min cycles", "median cycles", "Gelements/s");
    printf("  %s\n", std::string(85, '-').c_str());

    const int sizes[] = {1024, 4096, 16384, 65536, 1 << 20};

    for (int n : sizes) {
        auto input  = make_aligned_array<float>(n);
        auto out_simd   = make_aligned_array<float>(n);
        auto out_scalar = make_aligned_array<float>(n);

        for (int i = 0; i < n; ++i) input[i] = (float)(i % 100) * 0.05f - 2.5f;

        char label_simd[64], label_scalar[64];
        snprintf(label_simd,   sizeof(label_simd),   "AVX2  GeLU n=%7d", n);
        snprintf(label_scalar, sizeof(label_scalar),  "Scalar GeLU n=%7d", n);

        // Use element-count as "flops" proxy (each element ~15 FMA ops)
        long long ops = 15LL * n;

        BenchmarkRecord scalar_record{};
        scalar_record.category = "activations";
        scalar_record.name = label_scalar;
        scalar_record.n = n;
        measure(label_scalar,
                [&]() { gelu_forward_scalar(input.get(), out_scalar.get(), n); },
                ops, warmup, reps, &scalar_record);
        records.push_back(std::move(scalar_record));

        BenchmarkRecord simd_record{};
        simd_record.category = "activations";
        simd_record.name = label_simd;
        simd_record.n = n;
        measure(label_simd,
                [&]() { gelu_forward_avx2(input.get(), out_simd.get(), n); },
                ops, warmup, reps, &simd_record);
        records.push_back(std::move(simd_record));

        // Numerical accuracy check: max relative error
        double max_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double ref = out_scalar[i];
            double got = out_simd[i];
            double err = std::abs(ref - got) / (std::abs(ref) + 1e-7);
            if (err > max_err) max_err = err;
        }
        printf("    Max relative error (SIMD vs scalar): %.2e\n", max_err);
    }
}

static void bench_alignment(std::vector<BenchmarkRecord>& records,
                             int warmup, int reps) {
    printf("\n=== Memory Alignment Overhead Test ===\n");
    // Compare aligned vs deliberately unaligned GEMM loads (simulates
    // the cache-line crossing penalty on misaligned data.

    int M = 256, N = 256, K = 256;
    long long flops = 2LL * M * N * K;

    // Aligned allocation
    auto A_aligned = make_aligned_array<float>(M * K + 16);
    auto B_aligned = make_aligned_array<float>(K * N + 16);
    auto C_aligned = make_aligned_array<float>(M * N + 16);

    // Misaligned: offset by 4 bytes (one float = breaks 32-byte AVX alignment)
    float* A_mis = A_aligned.get() + 1;   // +4 bytes offset → crosses AVX boundary
    float* B_mis = B_aligned.get() + 1;
    float* C_mis = C_aligned.get() + 1;

    printf("  Aligned ptr:   %p (aligned to 64B: %s)\n",
           (void*)A_aligned.get(),
           is_aligned(A_aligned.get()) ? "YES" : "NO");
    printf("  Misaligned ptr: %p (aligned to 64B: %s)\n",
           (void*)A_mis,
           is_aligned(A_mis) ? "YES" : "NO");

    BenchmarkRecord aligned_record{};
    aligned_record.category = "gemm_alignment";
    aligned_record.name = "Aligned   GEMM 256x256";
    aligned_record.M = M;
    aligned_record.N = N;
    aligned_record.K = K;
    measure("Aligned   GEMM 256x256",
            [&]() {
                simd_sgemm(M, N, K, 1.0f,
                           A_aligned.get(), K, B_aligned.get(), N,
                           0.0f, C_aligned.get(), N);
            },
            flops, warmup, reps, &aligned_record);
    records.push_back(std::move(aligned_record));

    BenchmarkRecord misaligned_record{};
    misaligned_record.category = "gemm_alignment";
    misaligned_record.name = "Misaligned GEMM 256x256";
    misaligned_record.M = M;
    misaligned_record.N = N;
    misaligned_record.K = K;
    measure("Misaligned GEMM 256x256",
            [&]() {
                simd_sgemm(M, N, K, 1.0f,
                           A_mis, K, B_mis, N,
                           0.0f, C_mis, N);
            },
            flops, warmup, reps, &misaligned_record);
    records.push_back(std::move(misaligned_record));
}

// ─── Performance Summary Table ────────────────────────────────────────────────
static void print_roofline_summary() {
    printf("\n=== Roofline Model Summary (Intel Core, ~3.5 GHz, AVX2) ===\n");
    printf("  %-40s %s\n", "Parameter", "Value");
    printf("  %s\n", std::string(60, '-').c_str());
    printf("  %-40s %.0f GFLOPS\n", "Peak FP32 throughput (1 core, FMA):", 56.0);
    printf("  %-40s %.0f GB/s\n",   "Peak memory bandwidth (DDR5-4800):", 76.8);
    printf("  %-40s %.1f FLOP/byte\n", "Arithmetic intensity (256x256 GEMM):",
           (double)(2LL*256*256*256) / (3.0*256*256*4));
    printf("  %-40s %s\n", "Kernel regime:", "Compute-bound (AI > ridge point)");
    printf("  %-40s %s\n", "Bottleneck (small matrices < 64x64):", "Memory-bound (need tiling)");

    printf("\n  Performance Matrix:\n");
    printf("  %-16s %-22s %-22s %-10s\n",
           "Matrix Size", "Scalar -O3 (MCycles)", "SIMD AVX2 (MCycles)", "Speedup");
    printint argc, char** argv) {
    std::string json_output;
    int warmup = 3;
    int reps = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_output = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--json path] [--warmup n] [--reps n]\n", argv[0]);
            printf("  --json PATH   Write benchmark results in JSON format.\n");
            printf("  --warmup N    Number of warmup iterations (default 3).\n");
            printf("  --reps N      Number of measured repetitions (default 10).\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  SIMD-ML-Microkernels  ·  Cycle-Accurate Bench  ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  Measurement: RDTSC with LFENCE/RDTSCP serialisation\n");
    printf("  Warmup: %d reps · Measurement: %d reps · Metric: min(cycles)\n", warmup, reps);

#ifdef __linux__
    pin_to_core(0);
#endif
    g_tsc_hz = static_cast<double>(calibrate_tsc_hz());
    if (g_tsc_hz == 0.0) {
        fprintf(stderr, "Failed to calibrate TSC frequency; aborting.\n");
        return 1;
    }
    printf("  Calibrated TSC frequency: %.3f GHz\n", g_tsc_hz / 1e9);

    std::vector<BenchmarkRecord> records;
    bench_alignment(records, warmup, reps);
    bench_gemm(records, warmup, reps);
    bench_gelu(records, warmup, reps);
    print_roofline_summary();

    if (!json_output.empty()) {
        write_json_report(json_output, records);
    }

    return 0;
}
