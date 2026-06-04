FROM ubuntu:22.04

LABEL maintainer="Min Htet Myet"
LABEL description="IntrinsicML SIMD microkernel development environment"

ARG GCC_VERSION=12
ARG CMAKE_VERSION=3.28.0

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Pin all package versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-${GCC_VERSION}=12.3.0-1ubuntu1~22.04 \
    g++-${GCC_VERSION}=12.3.0-1ubuntu1~22.04 \
    python3.11=3.11.0~rc1-1~22.04 \
    python3.11-dev=3.11.0~rc1-1~22.04 \
    python3-pip \
    ninja-build \
    libopenblas-dev \
    numactl \
    linux-tools-generic \     # for perf
    && rm -rf /var/lib/apt/lists/*

# Install exact CMake version
RUN pip install cmake==${CMAKE_VERSION}

# pybind11 via pip (version-pinned)
RUN pip install pybind11==2.11.1 numpy==1.26.0 pytest==7.4.0

# Set GCC as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100

WORKDIR /workspace
COPY . .

# Build with all features
RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIMD_ML_OPENMP=ON \
    -DBENCH_OPENBLAS=ON \
    && cmake --build build --parallel

# Default: run tests
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]

# To run benchmarks:
# docker run --rm intrinsicml bash benchmarks/run_suite.sh