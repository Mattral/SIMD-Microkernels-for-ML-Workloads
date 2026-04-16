#pragma once

// Lightweight CPUID feature detection for runtime dispatch.
// See feedback.md BLOCK 2: runtime ISA dispatcher.

#include <cstdint>
#include <array>

namespace simd_ml {
namespace dispatch {

struct CpuFeatures {
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_fma = false;
    bool has_sse42 = false;

    static CpuFeatures detect() noexcept;
};

} // namespace dispatch
} // namespace simd_ml
