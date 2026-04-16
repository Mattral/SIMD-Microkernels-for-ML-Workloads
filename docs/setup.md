# Setup

## Prerequisites

Required packages on Linux:

* `cmake` (3.18+ recommended)
* `gcc` or `clang` with AVX2/FMA support
* `python3` and `pip`
* `pybind11` headers (installed via package manager or pip)
* `ninja` or `make` for build execution

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake python3-dev python3-pip git
pip install pybind11
```

## Build the C++ benchmark

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMD_ML_OPENMP=ON
cmake --build . --parallel
```

Run the benchmark executable:

```bash
./bench --json ../benchmarks/results/bench_results.json
```

If you need portable CMake builds without OpenMP:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMD_ML_OPENMP=OFF
cmake --build . --parallel
```

## Build and run tests

Within `build/`:

```bash
ctest -R simd_tests --output-on-failure
ctest -R simd_tests_doctest --output-on-failure
```

## Python extension

Install the Python package in editable mode:

```bash
cd /workspaces/SIMD-Microkernels-for-ML-Workloads
pip install -e .
```

Verify the Python API:

```bash
python -c "import simd_kernels; print(simd_kernels.build_info())"
```

## Benchmark automation

The repo includes a benchmark automation script:

```bash
./benchmarks/run_bench.sh
```

It builds the benchmark and writes a JSON report to `benchmarks/results/bench_results.json`.

## Recommended workflow

1. Build in `Release` mode.
2. Enable `SIMD_ML_OPENMP` when exercising threaded GEMM.
3. Use the benchmark JSON output for deterministic comparison.
4. Run unit tests after any kernel or API changes.
