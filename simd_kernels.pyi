"""
Type stubs for simd_kernels — IntrinsicML Python extension.

All kernels operate on float32 NumPy arrays and release the GIL during
computation. Arrays must be C-contiguous; call np.ascontiguousarray() if needed.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


# ─── Linear Algebra ───────────────────────────────────────────────────────────

def sgemm(
    A: NDArray[np.float32],
    B: NDArray[np.float32],
    C: Optional[NDArray[np.float32]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> NDArray[np.float32]:
    """
    SIMD GEMM: C = alpha * A @ B + beta * C  (float32, in-place or allocates).

    Parameters
    ----------
    A     : ndarray[float32], shape [M, K]
    B     : ndarray[float32], shape [K, N]
    C     : ndarray[float32], shape [M, N], optional (allocated if None)
    alpha : float scalar, default 1.0
    beta  : float scalar, default 0.0 (0 = overwrite C)

    Returns
    -------
    ndarray[float32], shape [M, N]
    """
    ...


# ─── Activation Functions ─────────────────────────────────────────────────────

def gelu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Out-of-place AVX2 GeLU (tanh approximation, max error < 5e-6).
    Returns a new array; input x is unchanged.
    """
    ...


def gelu_inplace(x: NDArray[np.float32]) -> None:
    """
    In-place AVX2 GeLU: x[:] = GeLU(x).
    Modifies x directly; returns None.
    """
    ...


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ReLU(x) = max(0, x) computed with AVX2 vectorization."""
    ...


def silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return SiLU(x) = x * sigmoid(x) using AVX2 vectorization."""
    ...


def softmax(
    x: NDArray[np.float32],
    axis: int = -1,
) -> NDArray[np.float32]:
    """Return numerically stable softmax over the specified axis (default: last)."""
    ...


def layer_norm(
    x: NDArray[np.float32],
    gamma: Optional[NDArray[np.float32]] = None,
    beta: Optional[NDArray[np.float32]] = None,
    eps: float = 1e-5,
) -> NDArray[np.float32]:
    """
    Layer Normalization over the last axis.

    output = (x - mean) / sqrt(var + eps) * gamma + beta

    Parameters
    ----------
    x     : ndarray[float32], any shape
    gamma : ndarray[float32], shape [last_dim], optional scale
    beta  : ndarray[float32], shape [last_dim], optional shift
    eps   : float, numerical stability (default 1e-5)
    """
    ...


# ─── Threading ────────────────────────────────────────────────────────────────

def set_num_threads(n: int) -> None:
    """Set the number of OpenMP threads used by GEMM (requires SIMD_ML_OPENMP build)."""
    ...


def get_num_threads() -> int:
    """Return the current OpenMP thread count used by GEMM."""
    ...


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def build_info() -> dict[str, str | bool]:
    """Return a dict with compiled ISA, FMA support, alignment, and build timestamp."""
    ...


def detected_isa() -> str:
    """Return the runtime-detected ISA label: 'avx512', 'avx2', 'sse42', or 'scalar'."""
    ...


def is_aligned(arr: np.ndarray) -> bool:
    """Return True if the array's data buffer is 64-byte aligned."""
    ...
