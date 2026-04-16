/**
 * test_threads.cpp — Verify OpenMP thread control and fallback behavior.
 */

#include "../src/kernels/gemm/avx2_gemm_packed.hpp"

int run_thread_tests() {
    int default_threads = simd_ml::gemm::get_num_threads();
    simd_ml::gemm::set_num_threads(1);
    bool pass = (simd_ml::gemm::get_num_threads() == 1);

#ifdef _OPENMP
    simd_ml::gemm::set_num_threads(2);
    int thread_count = simd_ml::gemm::get_num_threads();
    pass &= (thread_count == 2 || thread_count == 1);
#endif

    simd_ml::gemm::set_num_threads(default_threads);
    return pass ? 0 : 1;
}
