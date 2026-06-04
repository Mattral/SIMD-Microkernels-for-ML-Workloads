import numpy as np
import pytest

try:
    import simd_kernels
except ImportError:
    pytest.skip("simd_kernels not built — run cmake/pip install first", allow_module_level=True)


def test_sgemm_rejects_non_contiguous_slice():
    A = np.arange(16, dtype=np.float32).reshape(4, 4)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    C = np.zeros((4, 4), dtype=np.float32)

    A_slice = A[:, ::2]  # non-contiguous view

    with pytest.raises(RuntimeError, match="Input array must be C-contiguous"):
        simd_kernels.sgemm(A_slice, B, C)
