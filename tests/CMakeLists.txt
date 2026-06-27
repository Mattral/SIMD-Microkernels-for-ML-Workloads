# tests/CMakeLists.txt — IntrinsicML test build targets
#
# Link strategy (resolves the ODR violation from compiling avx_matmul.cpp twice):
#   simd_kernels_lib        (perf build: -O3 -ffast-math)  → provides simd_sgemm, scalar_sgemm
#   simd_kernels_precision_lib (-O2, no -ffast-math)       → provides all activation kernels
#
# simd_tests links BOTH; simd_kernels_lib is listed first so its simd_sgemm symbol wins.
# avx_matmul.cpp is NOT compiled into precision_lib (it only belongs in the perf lib).

# ─── doctest dependency ───────────────────────────────────────────────────────
find_package(doctest CONFIG QUIET)
if(NOT doctest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.12
        # EXCLUDE_FROM_ALL: prevents doctest's own install() rules from running
        # when the user runs `cmake --install`. Only our project artifacts install.
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(doctest)
endif()

# ─── Precision kernel library ─────────────────────────────────────────────────
# Compiled WITHOUT -ffast-math for accurate numerical validation.
# Excludes avx_matmul.cpp to prevent the ODR violation.
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

# ─── Main C++ test runner ─────────────────────────────────────────────────────
add_executable(simd_tests
    test_runner.cpp
    test_alignment.cpp
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
    test_activations.cpp
    test_threads.cpp
)
target_include_directories(simd_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
# Link order: perf lib first (provides simd_sgemm), then precision lib.
target_link_libraries(simd_tests PRIVATE simd_kernels_lib simd_kernels_precision_lib)
target_compile_options(simd_tests PRIVATE ${PRECISION_FLAGS})
add_test(NAME simd_tests COMMAND simd_tests)

# ─── AddressSanitizer / UBSan suite (optional, Clang/GCC only) ───────────────
if(SIMD_ML_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(simd_tests_asan
        test_runner.cpp
        test_alignment.cpp
        test_gelu.cpp
        test_gemm.cpp
        test_gemm_packed.cpp
        test_activations.cpp
        test_threads.cpp
    )
    target_include_directories(simd_tests_asan PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(simd_tests_asan PRIVATE simd_kernels_lib simd_kernels_precision_lib)
    target_compile_options(simd_tests_asan PRIVATE
        ${PRECISION_FLAGS}
        -fsanitize=address,undefined
        -fno-omit-frame-pointer)
    target_link_options(simd_tests_asan PRIVATE
        -fsanitize=address,undefined)
    add_test(NAME simd_tests_asan COMMAND simd_tests_asan)
    message(STATUS "ASan/UBSan test target enabled: simd_tests_asan")
endif()
