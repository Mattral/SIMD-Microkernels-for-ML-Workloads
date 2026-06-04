# tests/CMakeLists.txt — Build targets for precision and correctness tests

option(SIMD_ML_SANITIZE "Enable AddressSanitizer/UndefinedBehaviorSanitizer for test binaries" OFF)

# Precision test library: same kernel sources, compiled without -ffast-math
add_library(simd_kernels_precision_lib STATIC
    ../src/kernels/avx_matmul.cpp
    ../src/kernels/intrinsic_gelu.cpp
    ../src/kernels/gemm/avx2_gemm_packed.cpp
)
target_include_directories(simd_kernels_precision_lib PUBLIC ${CMAKE_SOURCE_DIR}/src/kernels)
target_compile_options(simd_kernels_precision_lib PRIVATE ${PRECISION_FLAGS})

target_compile_definitions(simd_kernels_precision_lib PRIVATE PRECISION_TEST_BUILD=1)

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

# Legacy test harness executable for manual correctness validation
add_executable(simd_tests
    test_runner.cpp
    test_alignment.cpp
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
)
target_include_directories(simd_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels)
target_link_libraries(simd_tests PRIVATE simd_kernels_precision_lib)
target_compile_options(simd_tests PRIVATE ${PRECISION_FLAGS})
add_test(NAME simd_tests COMMAND simd_tests)

# Doctest-based regression target
add_executable(simd_tests_doctest
    test_doctest.cpp
    test_threads.cpp
    test_alignment.cpp
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
)
target_include_directories(simd_tests_doctest PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels)
target_link_libraries(simd_tests_doctest PRIVATE simd_kernels_precision_lib doctest::doctest)
target_compile_options(simd_tests_doctest PRIVATE ${PRECISION_FLAGS})
add_test(NAME simd_tests_doctest COMMAND simd_tests_doctest)

if(SIMD_ML_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(simd_tests_asan
        test_doctest.cpp
        test_threads.cpp
    )
    target_include_directories(simd_tests_asan PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels)
    target_link_libraries(simd_tests_asan PRIVATE simd_kernels_precision_lib doctest::doctest)
    target_compile_options(simd_tests_asan PRIVATE ${PRECISION_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(simd_tests_asan PRIVATE -fsanitize=address,undefined)
    add_test(NAME simd_tests_asan COMMAND simd_tests_asan)
endif()
