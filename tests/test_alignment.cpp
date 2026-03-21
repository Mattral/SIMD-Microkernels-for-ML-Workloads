/**
 * test_alignment.cpp — Verify the aligned allocator meets contract.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include "../src/kernels/cache_alloc.hpp"

int run_alignment_tests() {
    printf("\n── Alignment Tests ──\n");
    bool all_pass = true;

    // Test 1: make_aligned_array produces 64-byte aligned pointers
    for (int n : {1, 7, 64, 1000, 65536}) {
        auto buf = make_aligned_array<float>(n);
        bool aligned = is_aligned(buf.get(), 64);
        printf("  make_aligned_array<%d>: ptr=%p aligned64=%s\n",
               n, (void*)buf.get(), aligned ? "YES" : "NO");
        all_pass &= aligned;
    }

    // Test 2: AlignedAllocator works with std::vector
    {
        std::vector<float, AlignedAllocator<float>> v(1024);
        bool aligned = is_aligned(v.data(), 64);
        printf("  std::vector<float, AlignedAllocator>: ptr=%p aligned64=%s\n",
               (void*)v.data(), aligned ? "YES" : "NO");
        all_pass &= aligned;
    }

    // Test 3: Misaligned raw pointer fails the check
    {
        auto buf = make_aligned_array<float>(64);
        float* misaligned = buf.get() + 1;  // +4 bytes = off alignment
        bool is_mis = !is_aligned(misaligned, 64);
        printf("  Misaligned ptr detection: %s\n", is_mis ? "PASS" : "FAIL");
        all_pass &= is_mis;
    }

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}

// ─── Forward declarations ────────────────────────────────────────────────────
int run_gemm_tests();
int run_gelu_tests();

// ─── Main test runner ─────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║  SIMD-ML-Microkernels  ·  Test Suite ║\n");
    printf("╚══════════════════════════════════════╝\n");

    int result = 0;
    result |= run_alignment_tests();
    result |= run_gelu_tests();
    result |= run_gemm_tests();

    printf("\n%s\n", result == 0
        ? "═══ ALL TESTS PASSED ═══"
        : "═══ SOME TESTS FAILED ═══");
    return result;
}
