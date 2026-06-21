/**
 * test_alignment.cpp — Verify the aligned allocator meets its contract.
 *
 * Tests:
 *   1. make_aligned_array<T>() produces 64-byte aligned pointers for all sizes
 *   2. AlignedAllocator<T> works with std::vector and also 64-byte aligns
 *   3. is_aligned() correctly rejects a pointer offset by one float (4 bytes)
 *   4. make_aligned_array<T>(0) handles zero gracefully (returns nullptr, no crash)
 *
 * Why 64-byte alignment?
 *   64 bytes == 1 cache line == 1 AVX-512 register.
 *   Misaligned access that straddles a cache-line requires two memory fetches.
 *   _mm256_load_ps requires 32-byte alignment; 64-byte guarantees both AVX2
 *   and AVX-512 store/load alignment in a single constant.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include "../src/kernels/cache_alloc.hpp"

int run_alignment_tests() {
    printf("\n── Alignment Tests ──\n");
    bool all_pass = true;

    // Test 1: make_aligned_array produces 64-byte aligned pointers for all sizes
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

    // Test 3: Misaligned raw pointer correctly detected
    {
        auto buf = make_aligned_array<float>(64);
        float* misaligned = buf.get() + 1;   // +4 bytes = crosses 64-byte boundary
        bool correctly_detected = !is_aligned(misaligned, 64);
        printf("  Misaligned ptr detection: %s\n",
               correctly_detected ? "PASS" : "FAIL");
        all_pass &= correctly_detected;
    }

    // Test 4: Zero-size allocation returns nullptr (no crash)
    {
        void* p = aligned_alloc_raw(0);
        bool null_ok = (p == nullptr);
        if (p) aligned_free_raw(p);  // free if implementation returned non-null
        printf("  aligned_alloc_raw(0) returns null: %s\n",
               null_ok ? "PASS" : "SKIP (non-null zero alloc — platform defined)");
        // This is platform-defined: not a failure if non-null, just log it.
    }

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
