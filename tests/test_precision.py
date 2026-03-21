"""
tests/test_precision.py — Numerical Precision vs NumPy/PyTorch

Validates that the SIMD GEMM and GeLU kernels agree with reference
implementations to within acceptable FP32 rounding tolerance.

Run with:
    pytest tests/test_precision.py -v

Requirements:
    pip install numpy pytest torch (optional)
    # Build the extension first:
    pip install -e .  # or: cmake --build build && cp build/simd_kernels*.so .
"""

import numpy as np
import pytest

try:
    import simd_kernels
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    pytest.skip("simd_kernels not built — run cmake/pip install first",
                allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


RNG = np.random.default_rng(seed=42)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rand_matrix(rows, cols, dtype=np.float32):
    return RNG.standard_normal((rows, cols)).astype(dtype)


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """Reference GeLU using tanh approximation (matches kernel formula)."""
    from numpy import tanh, sqrt, pi
    c = sqrt(2.0 / pi)
    return (0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x**3)))).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GEMM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGEMM:
    """Verify simd_kernels.sgemm matches np.dot to within FP32 tolerance."""

    GEMM_TOL = 1e-3   # FMA re-association with -ffast-math may reorder sums

    @pytest.mark.parametrize("M,N,K", [
        (8,   8,   8),
        (16,  16,  16),
        (64,  64,  64),
        (128, 128, 128),
        (256, 256, 256),
        (33,  17,  5),    # non-power-of-2
        (1,   1,   1),
    ])
    def test_sgemm_vs_numpy(self, M, N, K):
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C = np.zeros((M, N), dtype=np.float32)

        simd_kernels.sgemm(A, B, C)
        C_ref = (A @ B).astype(np.float32)

        max_err = float(np.max(np.abs(C - C_ref) / (np.abs(C_ref) + 1e-7)))
        assert max_err < self.GEMM_TOL, (
            f"GEMM [{M}×{N}×{K}]: max_rel_err={max_err:.2e} > tol={self.GEMM_TOL}"
        )

    def test_sgemm_alpha_beta(self):
        """Verify alpha scaling and beta accumulation."""
        M, N, K = 64, 64, 64
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C_init = rand_matrix(M, N)

        alpha, beta = 2.5, 0.3

        C_simd = C_init.copy()
        simd_kernels.sgemm(A, B, C_simd, alpha=alpha, beta=beta)

        C_ref = (alpha * (A @ B) + beta * C_init).astype(np.float32)

        max_err = float(np.max(np.abs(C_simd - C_ref) / (np.abs(C_ref) + 1e-7)))
        assert max_err < self.GEMM_TOL, f"alpha/beta: max_rel_err={max_err:.2e}"

    def test_sgemm_overwrites_with_beta_zero(self):
        """beta=0 must completely overwrite C (not += garbage)."""
        A = rand_matrix(32, 32)
        B = rand_matrix(32, 32)
        C = np.full((32, 32), 1e30, dtype=np.float32)  # poison value

        simd_kernels.sgemm(A, B, C, beta=0.0)
        C_ref = (A @ B).astype(np.float32)

        assert np.allclose(C, C_ref, rtol=1e-3, atol=1e-4), \
            "beta=0 should overwrite C entirely"


# ─────────────────────────────────────────────────────────────────────────────
# GeLU tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeLU:
    """Verify SIMD GeLU matches the numpy reference formula."""

    GELU_TOL = 5e-5   # polynomial approximation error budget

    @pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 1024, 65536])
    def test_gelu_vs_numpy(self, n):
        x = RNG.uniform(-3.0, 3.0, size=n).astype(np.float32)
        ref = gelu_numpy(x)

        out = simd_kernels.gelu(x)
        max_err = float(np.max(np.abs(out - ref) / (np.abs(ref) + 1e-7)))

        assert max_err < self.GELU_TOL, (
            f"GeLU n={n}: max_rel_err={max_err:.2e} > tol={self.GELU_TOL}"
        )

    def test_gelu_inplace_equals_outofplace(self):
        x = RNG.standard_normal(8192).astype(np.float32)
        out_oop = simd_kernels.gelu(x.copy())

        x_ip = x.copy()
        simd_kernels.gelu_inplace(x_ip)

        assert np.allclose(out_oop, x_ip, rtol=1e-6), \
            "in-place and out-of-place GeLU disagree"

    def test_gelu_negative_saturation(self):
        """For x ≪ 0, GeLU(x) → 0."""
        x = np.full(64, -10.0, dtype=np.float32)
        out = simd_kernels.gelu(x)
        assert np.allclose(out, 0.0, atol=1e-5), \
            "GeLU(-10) should be ≈ 0"

    def test_gelu_positive_passthrough(self):
        """For large x > 0, GeLU(x) ≈ x."""
        x = np.full(64, 10.0, dtype=np.float32)
        out = simd_kernels.gelu(x)
        assert np.allclose(out, x, rtol=1e-4), \
            "GeLU(10) should be ≈ 10"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_gelu_vs_pytorch(self):
        """Cross-check against torch.nn.functional.gelu (approximate=True)."""
        import torch
        import torch.nn.functional as F

        x_np = RNG.standard_normal(4096).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        ref = F.gelu(x_pt, approximate="tanh").numpy()
        got = simd_kernels.gelu(x_np)

        max_err = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-7)))
        assert max_err < self.GELU_TOL, \
            f"vs PyTorch GeLU: max_rel_err={max_err:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Alignment tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignment:
    """Verify that array alignment detection works correctly."""

    def test_numpy_array_alignment_check(self):
        """Standard np.zeros may or may not be 64-byte aligned; test detection."""
        x = np.zeros(1024, dtype=np.float32)
        result = simd_kernels.is_aligned(x)
        # We just check the function returns a bool — alignment depends on allocator
        assert isinstance(result, bool)

    def test_build_info_contains_isa(self):
        info = simd_kernels.build_info()
        assert "ISA" in info, "build_info should mention ISA"
        assert any(kw in info for kw in ("AVX", "Scalar")), \
            "build_info should mention AVX or Scalar"
