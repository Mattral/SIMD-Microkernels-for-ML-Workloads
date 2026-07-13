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
    isa: str = "",
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
    isa   : str, default "" — forces this call to use a specific kernel:
            "" (auto-detect, default), "avx2", or "avx512". Requesting
            "avx512" on hardware without AVX-512F+DQ raises RuntimeError
            (check avx512_available() first). "scalar" is a recognized
            value but raises RuntimeError ("not yet implemented") rather
            than silently falling back. This override applies only to
            this call, not subsequent ones.

    Returns
    -------
    ndarray[float32], shape [M, N]
    """
    ...


class GEMMConfig:
    """
    Callable GEMM configuration object.

    Stores default alpha, beta, and isa values for repeated GEMM calls.
    Tile parameters (tile_m, tile_n, tile_k, mr, nr) are accepted for API
    completeness and future auto-tuning integration; they are currently
    informational only.

    Examples
    --------
    >>> gemm = GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
    >>> C = gemm(A, B)
    """

    alpha: float
    beta: float
    isa: str
    tile_m: int
    tile_n: int
    tile_k: int
    mr: int
    nr: int

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        isa: str = "",
        tile_m: int = 128,
        tile_n: int = 2048,
        tile_k: int = 256,
        mr: int = 8,
        nr: int = 8,
    ) -> None: ...

    def __call__(
        self,
        A: NDArray[np.float32],
        B: NDArray[np.float32],
        C: Optional[NDArray[np.float32]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> NDArray[np.float32]: ...

    def __repr__(self) -> str: ...


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


def avx512_available() -> bool:
    """
    Return True if this CPU supports AVX-512F+DQ (required by sgemm_packed's
    8x16 micro-kernel), independent of any isa= override on a given call.

    Check this before requesting isa='avx512' explicitly to avoid a
    RuntimeError on unsupported hardware.
    """
    ...


def is_aligned(arr: np.ndarray) -> bool:
    """Return True if the array's data buffer is 64-byte aligned."""
    ...
