cmake_minimum_required(VERSION 3.22)
project(simd_ml_microkernels CXX)

# ─── Compiler requirements ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find dependencies ────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# ─── Detect AVX-512 support ───────────────────────────────────────────────────
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_HAS_AVX512F)
check_cxx_compiler_flag("-mavx2"    COMPILER_HAS_AVX2)
check_cxx_compiler_flag("-mfma"     COMPILER_HAS_FMA)

# ─── Compiler flags ───────────────────────────────────────────────────────────
# -O3           : full optimisation (vectorisation, loop unrolling, inlining)
# -march=native : use all ISA extensions available on the build host CPU
# -mfma         : enable FMA3 instruction set (Haswell+)
# -ffast-math   : permit reassociation / non-IEEE behaviour for speed
#                 (appropriate for ML inference; training may need strict FP)
# -DNDEBUG      : disable assert() overhead in production

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(SIMD_FLAGS "-O3" "-march=native" "-mfma" "-ffast-math" "-DNDEBUG")
    if(COMPILER_HAS_AVX512F)
        list(APPEND SIMD_FLAGS "-mavx512f" "-mavx512dq")
        message(STATUS "AVX-512 detected and enabled.")
    elseif(COMPILER_HAS_AVX2)
        list(APPEND SIMD_FLAGS "-mavx2")
        message(STATUS "AVX2 detected and enabled.")
    else()
        message(WARNING "No AVX2/AVX-512 support detected. Performance will be limited.")
    endif()
elseif(MSVC)
    set(SIMD_FLAGS "/O2" "/arch:AVX2" "/DNDEBUG" "/fp:fast")
endif()

# ─── Kernel library (shared between Python extension and bench) ───────────────
add_library(simd_kernels_lib STATIC
    src/kernels/avx_matmul.cpp
    src/kernels/intrinsic_gelu.cpp
)
target_include_directories(simd_kernels_lib PUBLIC src/kernels)
target_compile_options(simd_kernels_lib PRIVATE ${SIMD_FLAGS})

# ─── Python extension module ──────────────────────────────────────────────────
pybind11_add_module(simd_kernels
    src/bindings/pybind_entry.cpp
)
target_link_libraries(simd_kernels PRIVATE simd_kernels_lib)
target_compile_options(simd_kernels PRIVATE ${SIMD_FLAGS})

# ─── Standalone benchmark executable ─────────────────────────────────────────
add_executable(bench
    src/main_bench.cpp
)
target_link_libraries(bench PRIVATE simd_kernels_lib)
target_include_directories(bench PRIVATE src)
target_compile_options(bench PRIVATE ${SIMD_FLAGS})

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
