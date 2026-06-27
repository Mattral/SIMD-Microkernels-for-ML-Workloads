cmake_minimum_required(VERSION 3.22)
project(simd_ml_microkernels VERSION 0.2.0 LANGUAGES CXX)

# Must be included before any use of CMAKE_INSTALL_INCLUDEDIR / LIBDIR / etc.
# (used in target_include_directories() below and in the install() rules further down).
include(GNUInstallDirs)

# ─── Compiler requirements ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find dependencies ────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11)
find_package(OpenMP OPTIONAL_COMPONENTS CXX)
option(SIMD_ML_OPENMP "Enable OpenMP multithreading in GEMM" OFF)
option(BENCH_OPENBLAS "Link against OpenBLAS for baseline comparison" OFF)
option(SIMD_ML_SANITIZE "Enable AddressSanitizer/UndefinedBehaviorSanitizer for test binaries" OFF)

# ─── Detect AVX support ───────────────────────────────────────────────────────
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_HAS_AVX512F)
check_cxx_compiler_flag("-mavx2"    COMPILER_HAS_AVX2)
check_cxx_compiler_flag("-mfma"     COMPILER_HAS_FMA)

# ─── Compiler flags ───────────────────────────────────────────────────────────
# -O3           : full optimisation (vectorisation, loop unrolling, inlining)
# -O2           : safe optimisation for IEEE-compliant precision tests
# -march=native : use all ISA extensions available on the build host CPU
# -mfma         : enable FMA3 instruction set (Haswell+)
# -ffast-math   : permit reassociation / non-IEEE behaviour for speed
#                 (appropriate for ML inference; training may need strict FP)
# -DNDEBUG      : disable assert() overhead in production

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -Wall -Wextra: enable broad warning set to catch real defects.
    # NOT -Werror: warning sets differ across GCC/Clang versions; -Werror
    #              would cause spurious CI failures on newer compilers.
    set(WARN_FLAGS "-Wall" "-Wextra" "-Wshadow" "-Wno-unused-parameter")
    set(PERF_FLAGS      "-O3" "-march=native" "-mfma" "-ffast-math" "-DNDEBUG" ${WARN_FLAGS})
    set(PRECISION_FLAGS "-O2" "-march=native" "-mfma" "-DNDEBUG"              ${WARN_FLAGS})
    if(COMPILER_HAS_AVX512F)
        message(STATUS "AVX-512 compiler support detected. Building for host ISA only.")
    elseif(COMPILER_HAS_AVX2)
        message(STATUS "AVX2 compiler support detected. Building for host ISA.")
    else()
        message(WARNING "No AVX2/AVX-512 compiler support detected. Building scalar-safe path.")
    endif()
elseif(MSVC)
    set(PERF_FLAGS "/O2" "/arch:AVX2" "/DNDEBUG" "/fp:fast")
    set(PRECISION_FLAGS "/O2" "/arch:AVX2" "/DNDEBUG" "/fp:precise")
endif()

# ─── Kernel library (shared between Python extension and bench) ───────────────
add_library(simd_kernels_lib STATIC
    src/kernels/avx_matmul.cpp
    src/kernels/intrinsic_gelu.cpp
    src/kernels/gemm/avx2_gemm_packed.cpp
    src/kernels/activations/relu_avx2.cpp
    src/kernels/activations/silu_avx2.cpp
    src/kernels/activations/softmax_avx2.cpp
    src/kernels/activations/layer_norm_avx2.cpp
    src/kernels/gemm/gemm_dispatcher.cpp
)
# BUILD_INTERFACE: in-source build uses src/ and src/kernels/ directly.
# INSTALL_INTERFACE: installed consumers use ${prefix}/include/ so that
# #include <simd_kernels/foo.hpp> resolves to ${prefix}/include/simd_kernels/foo.hpp
target_include_directories(simd_kernels_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/kernels>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(simd_kernels_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(simd_kernels_lib PRIVATE ${PERF_FLAGS})

if(OpenMP_CXX_FOUND AND SIMD_ML_OPENMP)
    message(STATUS "OpenMP C++ found; enabling multithreaded GEMM support")
    target_link_libraries(simd_kernels_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(simd_kernels_lib PUBLIC SIMD_ML_OPENMP)
else()
    message(STATUS "OpenMP support disabled or not found")
endif()

# Alias: simd_kernels::simd_kernels → simd_kernels_lib
# This is the canonical target name that downstream find_package() consumers link against.
# The _lib suffix is an internal CMake convention; the public API name has no suffix.
add_library(simd_kernels::simd_kernels ALIAS simd_kernels_lib)

# ─── Python extension module (optional) ───────────────────────────────────────
if(pybind11_FOUND)
    pybind11_add_module(simd_kernels
        src/bindings/pybind_entry.cpp
    )
    target_link_libraries(simd_kernels PRIVATE simd_kernels_lib)
    target_compile_options(simd_kernels PRIVATE ${PERF_FLAGS})

    # CRITICAL: scikit-build-core builds a pip wheel by running `cmake --install`
    # and packaging whatever ends up installed. Once this project added other
    # install() rules below (static library, headers, CMake config, bench
    # executables) for `cmake --install` / find_package() consumers, the wheel
    # stopped containing the actual importable .so — because there was no
    # install() rule for the extension module itself. That produced wheels
    # that installed successfully but failed `import simd_kernels` with
    # ModuleNotFoundError. DESTINATION "." places it at the wheel root so
    # `import simd_kernels` resolves it directly from site-packages.
    install(TARGETS simd_kernels
        LIBRARY DESTINATION "." COMPONENT PythonModule
        RUNTIME DESTINATION "." COMPONENT PythonModule
    )
else()
    message(STATUS "pybind11 not found: skipping Python extension build")
endif()

# ─── Standalone benchmark executable ─────────────────────────────────────────
add_executable(bench
    src/main_bench.cpp
)
target_link_libraries(bench PRIVATE simd_kernels_lib)
target_include_directories(bench PRIVATE src)
target_compile_options(bench PRIVATE ${PERF_FLAGS})

# ─── Statistical benchmark tool (extended harness) ──────────────────────────
add_executable(bench_stat
    src/bench_stat.cpp
)
target_link_libraries(bench_stat PRIVATE simd_kernels_lib)
target_include_directories(bench_stat PRIVATE src)
target_compile_options(bench_stat PRIVATE ${PERF_FLAGS})

# Optional OpenBLAS linkage for baseline comparisons.
# Applied to BOTH bench and bench_stat: bench_stat's JSON output is what
# feeds docs/BENCHMARKS.md and benchmarks/check_regression.py, so it needs
# the same comparison capability as the RDTSC harness.
if(BENCH_OPENBLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(bench PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(bench PRIVATE BENCH_WITH_OPENBLAS)
        target_link_libraries(bench_stat PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(bench_stat PRIVATE BENCH_WITH_OPENBLAS)
        message(STATUS "OpenBLAS found; baseline comparison enabled in bench and bench_stat")
    else()
        message(WARNING "BENCH_OPENBLAS=ON but BLAS not found; disabling")
    endif()
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ─── Install rules ────────────────────────────────────────────────────────────
# Enables:  cmake --install <build_dir> --prefix /usr/local
# Installs: static library, all public headers, bench/bench_stat, CMake package config.
# Python extension is installed via pip (scikit-build-core), not cmake --install.
#
# After installation, downstream projects can use:
#   find_package(simd_kernels REQUIRED)
#   target_link_libraries(my_target PRIVATE simd_kernels::simd_kernels)

# ── Static library ───────────────────────────────────────────────────────────
# Exclude the precision_lib (test-only) and Python extension from install.
# Export the target so find_package() consumers can link against it.
install(TARGETS simd_kernels_lib
    EXPORT  simd_kernels-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT Development
)

# ── Public headers ───────────────────────────────────────────────────────────
# Install kernel headers (avx_matmul.hpp, intrinsic_gelu.hpp, etc.)
# Headers install into ${prefix}/include/simd_kernels/ so that consumers can
# write:  #include <simd_kernels/gemm/avx2_gemm_packed.hpp>
# The INSTALL_INTERFACE include path is ${CMAKE_INSTALL_INCLUDEDIR} (see above),
# so the compiler looks for simd_kernels/... relative to that prefix — which is
# exactly where the install(DIRECTORY) puts the files.
install(DIRECTORY src/kernels/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simd_kernels"
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "gemm/gemm_dispatcher*" EXCLUDE
    PATTERN "gemm/naive_gemm*"      EXCLUDE
)
install(DIRECTORY src/dispatch/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simd_kernels"
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
)

# ── Benchmark executables ─────────────────────────────────────────────────────
# Install under libexec/simd_kernels so they don't clutter PATH.
install(TARGETS bench bench_stat
    RUNTIME DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/simd_kernels"
    COMPONENT Benchmarks
)

# ── CMake package config ──────────────────────────────────────────────────────
# Generates simd_kernels-config.cmake and simd_kernels-config-version.cmake
# so downstream CMake projects can use find_package(simd_kernels).
include(CMakePackageConfigHelpers)

set(SIMD_KERNELS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/simd_kernels"
    CACHE PATH "Directory for simd_kernels CMake package config files")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/simd_kernels-config-version.cmake"
    VERSION     "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/simd_kernels-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/simd_kernels-config.cmake"
    INSTALL_DESTINATION "${SIMD_KERNELS_INSTALL_CMAKEDIR}"
)

# Install the exported target definition (needed for find_package to work)
install(EXPORT simd_kernels-targets
    FILE        simd_kernels-targets.cmake
    NAMESPACE   simd_kernels::
    DESTINATION "${SIMD_KERNELS_INSTALL_CMAKEDIR}"
    COMPONENT   Development
)

# Install the config files themselves
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/simd_kernels-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/simd_kernels-config-version.cmake"
    DESTINATION "${SIMD_KERNELS_INSTALL_CMAKEDIR}"
    COMPONENT   Development
)

