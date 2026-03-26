# tests/CMakeLists.txt — Build targets for precision and correctness tests

# Precision test library: same kernel sources, compiled without -ffast-math
add_library(simd_kernels_precision_lib STATIC
    ../src/kernels/avx_matmul.cpp
    ../src/kernels/intrinsic_gelu.cpp
)
target_include_directories(simd_kernels_precision_lib PUBLIC ${CMAKE_SOURCE_DIR}/src/kernels)
target_compile_options(simd_kernels_precision_lib PRIVATE ${PRECISION_FLAGS})

target_compile_definitions(simd_kernels_precision_lib PRIVATE PRECISION_TEST_BUILD=1)

# Test harness executable for alignment, GeLU, and GEMM correctness
add_executable(simd_tests
    test_alignment.cpp
    test_gelu.cpp
    test_gemm.cpp
)
target_include_directories(simd_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels)
target_link_libraries(simd_tests PRIVATE simd_kernels_precision_lib)
target_compile_options(simd_tests PRIVATE ${PRECISION_FLAGS})

add_test(NAME simd_tests COMMAND simd_tests)
