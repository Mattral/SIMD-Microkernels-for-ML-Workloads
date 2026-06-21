# tests/CMakeLists.txt — IntrinsicML test build targets

# ─── doctest dependency ───────────────────────────────────────────────────────
find_package(doctest CONFIG QUIET)
if(NOT doctest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.12
    )
    FetchContent_MakeAvailable(doctest)
endif()

# ─── Precision kernel library ─────────────────────────────────────────────────
# Compiled with -O2 (no -ffast-math) for accurate numerical validation.
# Includes only activation kernels and packed GEMM.
# avx_matmul.cpp is excluded: it is tested via simd_kernels_lib (perf build)
# to avoid the ODR violation that arises from compiling it twice.
add_library(simd_kernels_precision_lib STATIC
    ../src/kernels/intrinsic_gelu.cpp
    ../src/kernels/gemm/avx2_gemm_packed.cpp
    ../src/kernels/activations/relu_avx2.cpp
    ../src/kernels/activations/silu_avx2.cpp
    ../src/kernels/activations/softmax_avx2.cpp
    ../src/kernels/activations/layer_norm_avx2.cpp
    ../src/kernels/gemm/gemm_dispatcher.cpp
)
target_include_directories(simd_kernels_precision_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(simd_kernels_precision_lib PRIVATE ${PRECISION_FLAGS})
target_compile_definitions(simd_kernels_precision_lib PRIVATE PRECISION_TEST_BUILD=1)

# ─── Main C++ test runner ─────────────────────────────────────────────────────
# Links BOTH libs:
#   - simd_kernels_lib       (perf build, -O3 -ffast-math) for simd_sgemm
#   - simd_kernels_precision_lib (-O2, no ffast-math)      for activations
# Link order: precision_lib first so its symbols are preferred for gelu/relu/etc;
# simd_kernels_lib provides simd_sgemm and scalar_sgemm.
add_executable(simd_tests
    test_runner.cpp
    test_alignment.cpp
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
    test_activations.cpp
)
target_include_directories(simd_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
# Link simd_kernels_lib first so its simd_sgemm/scalar_sgemm are found first.
# The ODR is now resolved: only simd_kernels_lib has avx_matmul.cpp compiled.
target_link_libraries(simd_tests PRIVATE simd_kernels_lib simd_kernels_precision_lib)
target_compile_options(simd_tests PRIVATE ${PRECISION_FLAGS})
add_test(NAME simd_tests COMMAND simd_tests)

# ─── AddressSanitizer / UBSan suite (optional) ───────────────────────────────
if(SIMD_ML_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(simd_tests_asan
        test_runner.cpp
        test_alignment.cpp
        test_gelu.cpp
        test_gemm.cpp
        test_gemm_packed.cpp
        test_activations.cpp
    )
    target_include_directories(simd_tests_asan PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(simd_tests_asan PRIVATE simd_kernels_lib simd_kernels_precision_lib)
    target_compile_options(simd_tests_asan PRIVATE
        ${PRECISION_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(simd_tests_asan PRIVATE -fsanitize=address,undefined)
    add_test(NAME simd_tests_asan COMMAND simd_tests_asan)
endif()
