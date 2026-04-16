// Runtime kernel dispatcher for GEMM and simple activations.
// See feedback.md BLOCK 2 for design and tests.

#include "../../dispatch/cpuid.hpp"
#include "../../dispatch/kernel_registry.hpp"
#include "naive_gemm.hpp"
#include "avx2_gemm_packed.hpp"

#include <mutex>

namespace {
    static std::once_flag g_registry_once;
    static simd_ml::dispatch::KernelRegistry g_registry;
}

// Implement CpuFeatures::detect()
namespace simd_ml { namespace dispatch {

inline void cpuid(int leaf, int subleaf, unsigned int& a, unsigned int& b, unsigned int& c, unsigned int& d) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    a = b = c = d = 0;
    unsigned int aa=0, bb=0, cc=0, dd=0;
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("cpuid"
                     : "=a"(aa), "=b"(bb), "=c"(cc), "=d"(dd)
                     : "a"(leaf), "c"(subleaf));
    a = aa; b = bb; c = cc; d = dd;
#endif
#else
    a=b=c=d=0;
#endif
}

CpuFeatures CpuFeatures::detect() noexcept {
    CpuFeatures f;
#if defined(__x86_64__) || defined(__i386__)
    unsigned int a,b,c,d;
    // leaf 1: feature bits in ECX/EDX
    cpuid(1, 0, a,b,c,d);
    f.has_fma = (c & (1u << 12)) != 0;
    f.has_sse42 = (c & (1u << 20)) != 0;

    // leaf 7 subleaf 0 for extended features (EBX)
    cpuid(7, 0, a,b,c,d);
    f.has_avx2 = (b & (1u << 5)) != 0;
    f.has_avx512f = (b & (1u << 16)) != 0;
    f.has_avx512dq = (b & (1u << 17)) != 0;
#endif
    return f;
}

} }

namespace simd_ml { namespace dispatch {

const KernelRegistry& get_kernels() noexcept {
    std::call_once(g_registry_once, []{
        CpuFeatures f = CpuFeatures::detect();
        if (f.has_avx512f && f.has_avx512dq) {
            // AVX-512 path not implemented yet; fall back to AVX2 if available
            if (f.has_avx2 && f.has_fma) {
                g_registry.sgemm = &gemm::sgemm_packed;
                g_registry.gelu = &activations::gelu_avx2;
                g_registry.isa_label = "avx2";
            } else {
                g_registry.sgemm = &gemm_ref::naive_sgemm;
                g_registry.gelu = nullptr;
                g_registry.isa_label = "scalar";
            }
        } else if (f.has_avx2 && f.has_fma) {
            g_registry.sgemm = &gemm::sgemm_packed;
            g_registry.gelu = &activations::gelu_avx2;
            g_registry.isa_label = "avx2";
        } else if (f.has_sse42) {
            g_registry.sgemm = &gemm_ref::naive_sgemm; // SSE path not implemented
            g_registry.gelu = nullptr;
            g_registry.isa_label = "sse42";
        } else {
            g_registry.sgemm = &gemm_ref::naive_sgemm;
            g_registry.gelu = nullptr;
            g_registry.isa_label = "scalar";
        }
    });
    return g_registry;
}

void sgemm(int M, int N, int K,
           float alpha, const float* A, int lda,
           const float* B, int ldb,
           float beta, float* C, int ldc) {
    const KernelRegistry& reg = get_kernels();
    if (reg.sgemm) {
        reg.sgemm(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
    }
}

const char* detected_isa() noexcept {
    return get_kernels().isa_label;
}

} }
