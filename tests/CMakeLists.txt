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
# On non-x86 hosts (Apple Silicon etc.) only gemm_dispatcher.cpp is compiled —
# the AVX2 activation/GEMM sources all include <immintrin.h> and won't compile.
if(SIMD_ML_ENABLE_X86)
    set(PRECISION_LIB_SRCS
        ../src/kernels/intrinsic_gelu.cpp
        ../src/kernels/gemm/avx2_gemm_packed.cpp
        ../src/kernels/activations/relu_avx2.cpp
        ../src/kernels/activations/silu_avx2.cpp
        ../src/kernels/activations/softmax_avx2.cpp
        ../src/kernels/activations/layer_norm_avx2.cpp
        ../src/kernels/gemm/gemm_dispatcher.cpp
    )
else()
    set(PRECISION_LIB_SRCS
        ../src/kernels/gemm/gemm_dispatcher.cpp
    )
endif()
add_library(simd_kernels_precision_lib STATIC ${PRECISION_LIB_SRCS})
target_include_directories(simd_kernels_precision_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(simd_kernels_precision_lib PRIVATE ${PRECISION_FLAGS})

# ─── Main C++ test runner ─────────────────────────────────────────────────────
# On non-x86 hosts only the alignment test and a stub runner are compiled.
# All AVX2-specific test files (gelu, gemm, gemm_packed, activations, threads)
# include x86 intrinsic headers directly and must be excluded on ARM.
# test_runner.cpp uses SIMD_ML_X86_AVAILABLE to guard their call-sites.
set(TEST_SRCS_COMMON
    test_runner.cpp
    test_alignment.cpp
)
set(TEST_SRCS_X86
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
    test_gemm_f16.cpp
    test_gemm_bf16.cpp
    test_activations.cpp
    test_threads.cpp
)
if(SIMD_ML_ENABLE_X86)
    set(TEST_SRCS ${TEST_SRCS_COMMON} ${TEST_SRCS_X86})
else()
    set(TEST_SRCS ${TEST_SRCS_COMMON})
endif()
add_executable(simd_tests ${TEST_SRCS})
target_include_directories(simd_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
# Pass x86 availability as a preprocessor constant so test_runner.cpp can
# conditionally call run_gelu_tests(), run_gemm_tests(), etc.
if(SIMD_ML_ENABLE_X86)
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=1)
    # test_gemm_f16.cpp uses F16C intrinsics (helper converters); give it
    # the same -mf16c flag we pass to f16_gemm.cpp in the kernel library.
    if(COMPILER_HAS_F16C)
        set_source_files_properties(test_gemm_f16.cpp
            PROPERTIES COMPILE_OPTIONS "-mf16c")
    endif()
else()
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=0)
endif()
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
