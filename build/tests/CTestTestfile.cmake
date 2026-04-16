# CMake generated Testfile for 
# Source directory: /workspaces/SIMD-Microkernels-for-ML-Workloads/tests
# Build directory: /workspaces/SIMD-Microkernels-for-ML-Workloads/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[simd_tests]=] "/workspaces/SIMD-Microkernels-for-ML-Workloads/build/tests/simd_tests")
set_tests_properties([=[simd_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspaces/SIMD-Microkernels-for-ML-Workloads/tests/CMakeLists.txt;38;add_test;/workspaces/SIMD-Microkernels-for-ML-Workloads/tests/CMakeLists.txt;0;")
add_test([=[simd_tests_doctest]=] "/workspaces/SIMD-Microkernels-for-ML-Workloads/build/tests/simd_tests_doctest")
set_tests_properties([=[simd_tests_doctest]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspaces/SIMD-Microkernels-for-ML-Workloads/tests/CMakeLists.txt;52;add_test;/workspaces/SIMD-Microkernels-for-ML-Workloads/tests/CMakeLists.txt;0;")
subdirs("../_deps/doctest-build")
