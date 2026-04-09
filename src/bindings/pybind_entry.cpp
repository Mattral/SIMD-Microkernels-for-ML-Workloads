/**
 * pybind_entry.cpp — Python Bindings via pybind11
 *
 * Exposes the SIMD GEMM and GeLU kernels directly to Python with zero-copy
 * NumPy array access. No data is copied between Python and C++; the kernel
 * receives raw float* pointers directly into the NumPy buffer.
 *
 * ─── Zero-Copy Strategy ───────────────────────────────────────────────────────
 * pybind11's py::array_t<float, py::array::c_contiguous> descriptor enforces:
 *   1. dtype == float32
 *   2. C-contiguous memory layout (row-major)
 *   3. Direct access to the buffer pointer via .mutable_data() / .data()
 *
 * This means `np.array(..., dtype=np.float32)` passed from Python goes directly
 * to the C++ kernel with no allocation or copy overhead.
 *
 * ─── Build ────────────────────────────────────────────────────────────────────
 *   pip install pybind11 numpy
 *   # Or use pyproject.toml + scikit-build — see root pyproject.toml
 *
 *   # Manual compile (for development):
 *   g++ -O3 -march=native -mfma -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       src/bindings/pybind_entry.cpp \
 *       src/kernels/avx_matmul.cpp \
 *       src/kernels/intrinsic_gelu.cpp \
 *       -o simd_kernels$(python3-config --extension-suffix)
 *
 * ─── Python Usage ─────────────────────────────────────────────────────────────
 *   import numpy as np
 *   import simd_kernels
 *
 *   A = np.random.randn(256, 256).astype(np.float32)
 *   B = np.random.randn(256, 256).astype(np.float32)
 *   C = np.zeros((256, 256), dtype=np.float32)
 *
 *   simd_kernels.sgemm(A, B, C)            # C = A @ B  (SIMD, in-place)
 *
 *   x = np.random.randn(65536).astype(np.float32)
 *   simd_kernels.gelu_inplace(x)           # x[:] = GeLU(x)  (AVX2, in-place)
 *
 *   # Query what ISA was compiled in
 *   print(simd_kernels.build_info())
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../kernels/activations/activations.hpp"
#include "../kernels/intrinsic_gelu.hpp"
#include "../kernels/cache_alloc.hpp"
#include "../dispatch/kernel_registry.hpp"

namespace py = pybind11;

// ─── Helpers ──────────────────────────────────────────────────────────────────

static void check_float32(const py::array& arr, const char* name) {
    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 4) {
        throw std::runtime_error(std::string(name) + " must be float32 (dtype=np.float32)");
    }
}

static void check_contiguous(const py::array& arr, const char* name) {
    if (!arr.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::runtime_error(
            std::string("Input array must be C-contiguous. ")
            + "Call numpy.ascontiguousarray(arr) before passing.");
    }
}

static void validate_array(const py::array& arr, const char* name) {
    check_float32(arr, name);
    check_contiguous(arr, name);
}

// ─── Binding: sgemm ──────────────────────────────────────────────────────────
/**
 * sgemm(A, B, C, alpha=1.0, beta=0.0)
 * Computes C = alpha * A @ B + beta * C  (in-place, zero-copy).
 *
 * Args:
 *   A     : np.ndarray shape [M, K], dtype float32, C-contiguous
 *   B     : np.ndarray shape [K, N], dtype float32, C-contiguous
 *   C     : np.ndarray shape [M, N], dtype float32, C-contiguous, writeable
 *   alpha : float scalar (default 1.0)
 *   beta  : float scalar for C scaling (default 0.0 = overwrite)
 */
static void py_sgemm(py::array A,
                     py::array B,
                     py::array C,
                     float alpha = 1.0f,
                     float beta  = 0.0f) {
    validate_array(A, "A");
    validate_array(B, "B");
    validate_array(C, "C");

    if (A.ndim() != 2) throw std::runtime_error("A must be 2-D");
    if (B.ndim() != 2) throw std::runtime_error("B must be 2-D");
    if (C.ndim() != 2) throw std::runtime_error("C must be 2-D");

    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    if (B.shape(0) != K)
        throw std::invalid_argument("A.shape[1] must equal B.shape[0]");
    if (C.shape(0) != M || C.shape(1) != N)
        throw std::runtime_error("C must have shape [M, N]");

    bool A_is_aligned = is_aligned(A.data(), 32);
    bool B_is_aligned = is_aligned(B.data(), 32);
    bool C_is_aligned = is_aligned(C.mutable_data(), 32);
    if (!A_is_aligned || !B_is_aligned || !C_is_aligned) {
        // Fall back to unaligned-safe kernels. The inner AVX2 path uses
        // _mm256_loadu_ps/_mm256_storeu_ps where needed, so misaligned
        // NumPy buffers do not crash with SIGBUS.
    }

    // Release GIL during the (potentially long) GEMM computation
    {
        py::gil_scoped_release release;
        simd_ml::dispatch::sgemm(M, N, K,
                                 alpha,
                                 static_cast<const float*>(A.data()), K,
                                 static_cast<const float*>(B.data()), N,
                                 beta,
                                 static_cast<float*>(C.mutable_data()), N);
    }
}

// ─── Binding: gelu_inplace ────────────────────────────────────────────────────
/**
 * gelu_inplace(x)
 * Applies GeLU element-wise to `x` in-place. AVX2-vectorized.
 *
 * Args:
 *   x : np.ndarray (any shape), dtype float32, C-contiguous, writeable
 */
static void py_gelu_inplace(py::array x) {
    validate_array(x, "x");
    x.request(true);  // ensure writable buffer

    float* ptr = static_cast<float*>(x.mutable_data());
    std::size_t n = static_cast<std::size_t>(x.size());
    bool x_is_aligned = is_aligned(ptr, 32);
    (void)x_is_aligned;  // kernel already handles misaligned loads safely

    {
        py::gil_scoped_release release;
        gelu_forward_avx2(ptr, ptr, n);  // in-place: input == output
    }
}

// ─── Binding: gelu (out-of-place) ────────────────────────────────────────────
/**
 * gelu(x) -> np.ndarray
 * Returns a new array GeLU(x). Allocates output.
 */
static py::array_t<float> py_gelu(py::array x) {
    validate_array(x, "x");

    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);

    bool x_is_aligned = is_aligned(x.data(), 32);
    (void)x_is_aligned;

    {
        py::gil_scoped_release release;
        gelu_forward_avx2(static_cast<const float*>(x.data()), out.mutable_data(), n);
    }
    return out;
}

static py::array_t<float> py_relu(py::array x) {
    validate_array(x, "x");
    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);
    {
        py::gil_scoped_release release;
        activations::relu_avx2(static_cast<const float*>(x.data()), out.mutable_data(), static_cast<int>(n));
    }
    return out;
}

static py::array_t<float> py_silu(py::array x) {
    validate_array(x, "x");
    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);
    {
        py::gil_scoped_release release;
        activations::silu_avx2(static_cast<const float*>(x.data()), out.mutable_data(), static_cast<int>(n));
    }
    return out;
}

static py::array_t<float> py_softmax(py::array x, int axis = -1) {
    validate_array(x, "x");
    if (x.ndim() < 1) {
        throw std::runtime_error("softmax requires a non-empty array");
    }
    int ndim = x.ndim();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("softmax axis out of range");
    }
    if (axis != ndim - 1) {
        throw std::runtime_error("softmax currently supports the last axis only");
    }

    auto buf = x.request();
    py::array_t<float> out(buf.shape);
    float* out_ptr = out.mutable_data();
    const float* in_ptr = static_cast<const float*>(x.data());
    int row_size = static_cast<int>(buf.shape[ndim - 1]);
    int rows = 1;
    for (int i = 0; i < ndim - 1; ++i) {
        rows = static_cast<int>(rows * buf.shape[i]);
    }

    {
        py::gil_scoped_release release;
        for (int r = 0; r < rows; ++r) {
            const float* row_in = in_ptr + static_cast<std::size_t>(r) * row_size;
            float* row_out = out_ptr + static_cast<std::size_t>(r) * row_size;
            activations::softmax_row_avx2(row_in, row_out, row_size);
        }
    }
    return out;
}

static void py_set_num_threads(int n) {
    if (n <= 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
    simd_ml::gemm::set_num_threads(n);
}

static int py_get_num_threads() {
    return simd_ml::gemm::get_num_threads();
}

// ─── Binding: build_info ──────────────────────────────────────────────────────
static std::string build_info() {
    std::string info = "SIMD-ML-Microkernels build info:\n";
#ifdef __AVX512F__
    info += "  ISA:         AVX-512F + AVX-512DQ\n";
#elif defined(__AVX2__)
    info += "  ISA:         AVX2 + FMA3\n";
#else
    info += "  ISA:         Scalar fallback (no AVX2 detected)\n";
#endif
#ifdef __FMA__
    info += "  FMA:         enabled (_mm256_fmadd_ps)\n";
#endif
    info += "  Alignment:   64-byte (posix_memalign / _aligned_malloc)\n";
    info += "  Build:       " __DATE__ " " __TIME__ "\n";
    info += "  Runtime ISA: " + std::string(simd_ml::dispatch::detected_isa()) + "\n";
    return info;
}

// ─── Binding: is_aligned_ptr ─────────────────────────────────────────────────
static bool py_is_aligned(py::array arr) {
    return is_aligned(arr.data());
}

// ─── Module definition ────────────────────────────────────────────────────────
PYBIND11_MODULE(simd_kernels, m) {
    m.doc() =
        "SIMD-ML-Microkernels: Hand-vectorized AVX2/AVX-512 GEMM and GeLU "
        "exposed as Python extensions via pybind11. All operations are "
        "zero-copy: NumPy array buffers are used directly.";

    m.def("sgemm", &py_sgemm,
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          R"doc(
SIMD GEMM: computes C = alpha * A @ B + beta * C (in-place, float32).

Parameters
----------
A     : np.ndarray[float32], shape [M, K], C-contiguous
B     : np.ndarray[float32], shape [K, N], C-contiguous
C     : np.ndarray[float32], shape [M, N], C-contiguous, writeable
alpha : float, default 1.0
beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate)

Notes
-----
For best performance, pass 64-byte aligned arrays:
    A = np.ascontiguousarray(A, dtype=np.float32)
    # Allocated by NumPy, alignment not guaranteed — use aligned buffer
    # for peak throughput (the kernel handles unaligned via _mm256_loadu_ps).
)doc");

    m.def("gelu_inplace", &py_gelu_inplace,
          py::arg("x"),
          R"doc(
In-place AVX2-vectorized GeLU activation: x[:] = GeLU(x).

Uses the tanh approximation: GeLU(x) = 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
Polynomial evaluation is fully vectorized — no scalar branches.

Parameters
----------
x : np.ndarray[float32], any shape, writeable
)doc");

    m.def("gelu", &py_gelu,
          py::arg("x"),
          R"doc(
Out-of-place AVX2-vectorized GeLU activation.
Returns a new array GeLU(x).  Input x is not modified.
)doc");

    m.def("relu", &py_relu,
          py::arg("x"),
          "Return ReLU(x) computed with AVX2 vectorization.");

    m.def("silu", &py_silu,
          py::arg("x"),
          "Return SiLU(x) computed with AVX2 vectorization.");

    m.def("softmax", &py_softmax,
          py::arg("x"), py::arg("axis") = -1,
          "Return row-wise softmax over the last axis.");

    m.def("set_num_threads", &py_set_num_threads,
          py::arg("n"),
          "Set the number of OpenMP threads used by GEMM.");

    m.def("get_num_threads", &py_get_num_threads,
          "Return the current OpenMP thread count used by GEMM.");

    m.def("build_info", &build_info,
          "Return a string describing the compiled ISA and build configuration.");

    m.def("detected_isa", []() {
        // Return the runtime-dispatched ISA label selected by the kernel
        return std::string(simd_ml::dispatch::detected_isa());
    }, "Return the detected ISA label (avx512|avx2|sse42|scalar)");
    m.def("is_aligned", &py_is_aligned,
          py::arg("arr"),
          "Return True if the array's data buffer is 64-byte aligned.");
}
