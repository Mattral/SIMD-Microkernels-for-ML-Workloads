"""
tests/test_precision.py — Numerical Precision Tests vs NumPy/SciPy/PyTorch

Validates that all SIMD kernels agree with reference implementations to
within acceptable FP32 rounding tolerance.

Run with:
    pytest tests/test_precision.py -v

Requirements:
    pip install numpy pytest scipy  (torch is optional)
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
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.special import erf as scipy_erf
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


RNG = np.random.default_rng(seed=42)

# ─── Reference implementations ───────────────────────────────────────────────

def rand_matrix(rows, cols, dtype=np.float32):
    return RNG.standard_normal((rows, cols)).astype(dtype)

def rand_array(n, dtype=np.float32):
    return RNG.standard_normal(n).astype(dtype)

def gelu_tanh_numpy(x: np.ndarray) -> np.ndarray:
    """Reference GeLU using tanh approximation (matches SIMD kernel formula)."""
    x64 = x.astype(np.float64)
    c = np.sqrt(2.0 / np.pi)
    return (0.5 * x64 * (1.0 + np.tanh(c * (x64 + 0.044715 * x64**3)))).astype(np.float64)

def relu_numpy(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x.astype(np.float64))

def silu_numpy(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return x64 / (1.0 + np.exp(-x64))

def softmax_numpy(x: np.ndarray, axis=-1) -> np.ndarray:
    x64 = x.astype(np.float64)
    x64 -= x64.max(axis=axis, keepdims=True)
    e = np.exp(x64)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm_numpy(x: np.ndarray, gamma=None, beta=None, eps=1e-5) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var  = x64.var(axis=-1, keepdims=True)
    xn   = (x64 - mean) / np.sqrt(var + eps)
    if gamma is not None:
        xn = xn * gamma.astype(np.float64)
    if beta is not None:
        xn = xn + beta.astype(np.float64)
    return xn


# ─── GEMM Tests ──────────────────────────────────────────────────────────────

class TestGEMM:
    """Verify simd_kernels.sgemm matches np.dot to within FP32 tolerance."""

    GEMM_TOL = 5e-2  # -ffast-math allows FP reassociation; O(K*eps_mach) error   # FMA re-association with -ffast-math may reorder sums

    @pytest.mark.parametrize("M,N,K", [
        (1,   1,   1),
        (8,   8,   8),
        (16,  16,  16),
        (64,  64,  64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (33,  17,  5),    # non-power-of-2
        (1,   4096, 128), # skinny M (typical attention projection)
        (7,   7,   7),    # odd sizes
    ])
    def test_sgemm_vs_numpy(self, M, N, K):
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C = np.zeros((M, N), dtype=np.float32)

        simd_kernels.sgemm(A, B, C)
        C_ref = A.astype(np.float64) @ B.astype(np.float64)

        max_err = float(np.max(np.abs(C.astype(np.float64) - C_ref)
                               / (np.abs(C_ref) + 1e-7)))
        assert max_err < self.GEMM_TOL, (
            f"GEMM [{M}×{N}×{K}]: max_rel_err={max_err:.2e} > tol={self.GEMM_TOL}"
        )

    def test_sgemm_allocates_output_when_C_not_provided(self):
        A = rand_matrix(32, 64)
        B = rand_matrix(64, 32)
        C = simd_kernels.sgemm(A, B)
        assert C.shape == (32, 32)
        assert C.dtype == np.float32
        C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        assert np.allclose(C, C_ref, rtol=self.GEMM_TOL, atol=1e-4)

    def test_sgemm_alpha_beta(self):
        """Verify alpha scaling and beta accumulation."""
        M, N, K = 64, 64, 64
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C_init = rand_matrix(M, N)

        alpha, beta = 2.5, 0.3
        C_simd = C_init.copy()
        simd_kernels.sgemm(A, B, C_simd, alpha=alpha, beta=beta)

        C_ref = alpha * (A.astype(np.float64) @ B.astype(np.float64)) \
              + beta * C_init.astype(np.float64)
        max_err = float(np.max(np.abs(C_simd.astype(np.float64) - C_ref)
                               / (np.abs(C_ref) + 1e-7)))
        assert max_err < self.GEMM_TOL, f"alpha/beta: max_rel_err={max_err:.2e}"

    def test_sgemm_beta_zero_overwrites_C(self):
        """beta=0 must completely overwrite C — not accumulate into garbage."""
        A = rand_matrix(32, 32)
        B = rand_matrix(32, 32)
        C = np.full((32, 32), 1e30, dtype=np.float32)   # poison value
        simd_kernels.sgemm(A, B, C, beta=0.0)
        C_ref = (A @ B).astype(np.float32)
        assert np.allclose(C, C_ref, rtol=1e-3, atol=1e-4), \
            "beta=0 should overwrite C entirely"

    def test_sgemm_shape_mismatch_raises(self):
        A = rand_matrix(32, 64)
        B = rand_matrix(32, 32)  # wrong: K-dimension mismatch
        with pytest.raises(RuntimeError, match="Shape mismatch|ndim"):
            simd_kernels.sgemm(A, B)

    def test_sgemm_non_contiguous_raises(self):
        A = rand_matrix(32, 32)
        B = rand_matrix(32, 32)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            simd_kernels.sgemm(A[::2], B)   # strided = not C-contiguous


# ─── GeLU Tests ──────────────────────────────────────────────────────────────

class TestGeLU:
    """Verify SIMD GeLU matches the fast tanh approximation reference."""

    GELU_TOL = 1e-4  # Cody-Waite exp-tanh: < 2e-7 abs tanh error, amplified by GeLU scaling   # rational polynomial approximation error budget

    @pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 64, 1024, 65536])
    def test_gelu_vs_tanh_approx(self, n):
        x = RNG.uniform(-3.0, 3.0, size=n).astype(np.float32)
        ref = gelu_tanh_numpy(x)
        out = simd_kernels.gelu(x)
        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)
                               / (np.abs(ref) + 1e-7)))
        assert max_err < self.GELU_TOL, (
            f"GeLU n={n}: max_rel_err={max_err:.2e} > tol={self.GELU_TOL}"
        )

    def test_gelu_sweep_full_range(self):
        """Dense sweep over [-5, 5] where the polynomial approximation is valid."""
        x = np.arange(-5.0, 5.0 + 1e-12, 0.001, dtype=np.float32)
        ref = gelu_tanh_numpy(x)
        out = simd_kernels.gelu(x)
        max_abs_err = float(np.max(np.abs(out.astype(np.float64) - ref)))
        assert max_abs_err < 1e-4, (
            f"GeLU sweep [-5,5]: max_abs_err={max_abs_err:.2e} > 1e-4"
        )

    def test_gelu_inplace_equals_outofplace(self):
        x = RNG.standard_normal(8192).astype(np.float32)
        out_oop = simd_kernels.gelu(x.copy())
        x_ip = x.copy()
        simd_kernels.gelu_inplace(x_ip)
        assert np.allclose(out_oop, x_ip, rtol=1e-6), \
            "In-place and out-of-place GeLU must agree"

    def test_gelu_negative_saturation(self):
        """For x ≪ 0, GeLU(x) → 0."""
        x = np.full(64, -10.0, dtype=np.float32)
        out = simd_kernels.gelu(x)
        assert np.allclose(out, 0.0, atol=1e-5), "GeLU(-10) should be ≈ 0"

    def test_gelu_positive_passthrough(self):
        """For large x > 0, GeLU(x) ≈ x."""
        x = np.full(64, 10.0, dtype=np.float32)
        out = simd_kernels.gelu(x)
        assert np.allclose(out, x, rtol=1e-3), "GeLU(10) should be ≈ 10"

    def test_gelu_output_shape_preserved(self):
        for shape in [(8,), (4, 16), (2, 3, 32)]:
            x = rand_array(int(np.prod(shape))).reshape(shape)
            out = simd_kernels.gelu(x)
            assert out.shape == shape, f"shape {shape} not preserved"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_gelu_vs_pytorch_tanh_approx(self):
        """Cross-check against torch.nn.functional.gelu(approximate='tanh')."""
        x_np = RNG.standard_normal(4096).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        ref = F.gelu(x_pt, approximate="tanh").numpy()
        got = simd_kernels.gelu(x_np)
        max_err = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-7)))
        assert max_err < self.GELU_TOL, f"vs PyTorch GeLU: max_rel_err={max_err:.2e}"


# ─── ReLU Tests ──────────────────────────────────────────────────────────────

class TestReLU:
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 1024, 65537])
    def test_relu_vs_numpy(self, n):
        x = RNG.uniform(-3.0, 3.0, size=n).astype(np.float32)
        ref = relu_numpy(x).astype(np.float32)
        out = simd_kernels.relu(x)
        # ReLU is exact (just a max with zero) — should match exactly
        assert np.array_equal(out, ref), f"ReLU n={n}: mismatch"

    def test_relu_all_negative(self):
        x = np.full(64, -5.0, dtype=np.float32)
        assert np.all(simd_kernels.relu(x) == 0.0)

    def test_relu_all_positive(self):
        x = np.arange(1, 65, dtype=np.float32)
        out = simd_kernels.relu(x)
        assert np.array_equal(out, x)


# ─── SiLU Tests ──────────────────────────────────────────────────────────────

class TestSiLU:
    SILU_TOL = 1e-4

    @pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 1024, 65536])
    def test_silu_vs_numpy(self, n):
        x = RNG.uniform(-4.0, 4.0, size=n).astype(np.float32)
        ref = silu_numpy(x)
        out = simd_kernels.silu(x)
        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)
                               / (np.abs(ref) + 1e-7)))
        assert max_err < self.SILU_TOL, f"SiLU n={n}: max_rel_err={max_err:.2e}"

    def test_silu_zero_input(self):
        x = np.zeros(64, dtype=np.float32)
        out = simd_kernels.silu(x)
        assert np.allclose(out, 0.0, atol=1e-7)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_silu_vs_pytorch(self):
        x_np = RNG.standard_normal(4096).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        ref = F.silu(x_pt).numpy()
        got = simd_kernels.silu(x_np)
        max_err = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-7)))
        assert max_err < self.SILU_TOL, f"vs PyTorch SiLU: max_rel_err={max_err:.2e}"


# ─── Softmax Tests ────────────────────────────────────────────────────────────

class TestSoftmax:
    @pytest.mark.parametrize("shape", [(8,), (64,), (512,), (4, 128), (16, 64)])
    def test_softmax_sums_to_one(self, shape):
        x = RNG.standard_normal(shape).astype(np.float32)
        out = simd_kernels.softmax(x)
        row_sums = out.reshape(-1, shape[-1]).sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Softmax row sums not 1.0 for shape {shape}: {row_sums}"

    @pytest.mark.parametrize("n", [4, 16, 64, 512])
    def test_softmax_vs_numpy(self, n):
        x = RNG.standard_normal(n).astype(np.float32)
        ref = softmax_numpy(x)
        out = simd_kernels.softmax(x)
        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)))
        assert max_err < 1e-5, f"Softmax n={n}: max_abs_err={max_err:.2e}"

    def test_softmax_argmax_preserved(self):
        """argmax(softmax(x)) == argmax(x)."""
        x = RNG.standard_normal(256).astype(np.float32)
        out = simd_kernels.softmax(x)
        assert np.argmax(out) == np.argmax(x)

    def test_softmax_numerical_stability_large_values(self):
        """Input with very large values should not produce NaN/Inf."""
        x = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
        out = simd_kernels.softmax(x)
        assert np.all(np.isfinite(out)), "Softmax must be numerically stable for large inputs"
        assert np.allclose(out.sum(), 1.0, atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_softmax_vs_pytorch_2d(self):
        x = RNG.standard_normal((8, 64)).astype(np.float32)
        ref = F.softmax(torch.from_numpy(x), dim=-1).numpy()
        out = simd_kernels.softmax(x, axis=-1)
        assert np.allclose(out, ref, atol=1e-5)


# ─── LayerNorm Tests ──────────────────────────────────────────────────────────

class TestLayerNorm:
    LN_TOL = 1e-4

    @pytest.mark.parametrize("n", [8, 64, 128, 256, 512, 768, 1024])
    def test_layer_norm_zero_mean_unit_std(self, n):
        """Without gamma/beta, output should have mean≈0 and std≈1."""
        x = RNG.standard_normal(n).astype(np.float32)
        out = simd_kernels.layer_norm(x)
        assert abs(float(out.mean())) < 1e-4, \
            f"LayerNorm n={n}: mean={float(out.mean()):.4e} ≠ 0"
        assert abs(float(out.std()) - 1.0) < 1e-3, \
            f"LayerNorm n={n}: std={float(out.std()):.5f} ≠ 1"

    @pytest.mark.parametrize("n", [64, 256, 768])
    def test_layer_norm_with_affine(self, n):
        """With gamma/beta, output must match the numpy reference."""
        x     = RNG.standard_normal(n).astype(np.float32)
        gamma = (1.0 + 0.1 * RNG.standard_normal(n)).astype(np.float32)
        beta  = (0.01 * RNG.standard_normal(n)).astype(np.float32)

        ref = layer_norm_numpy(x, gamma, beta)
        out = simd_kernels.layer_norm(x, gamma=gamma, beta=beta)

        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)
                               / (np.abs(ref) + 1e-7)))
        assert max_err < self.LN_TOL, \
            f"LayerNorm affine n={n}: max_rel_err={max_err:.2e}"

    def test_layer_norm_2d_batch(self):
        """LayerNorm applied to each row of a 2D array."""
        x     = RNG.standard_normal((16, 128)).astype(np.float32)
        out   = simd_kernels.layer_norm(x)
        ref   = layer_norm_numpy(x)
        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)
                               / (np.abs(ref) + 1e-7)))
        assert max_err < self.LN_TOL, f"LayerNorm 2D: max_rel_err={max_err:.2e}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_layer_norm_vs_pytorch(self):
        n     = 256
        x     = RNG.standard_normal(n).astype(np.float32)
        gamma = np.ones(n, dtype=np.float32)
        beta  = np.zeros(n, dtype=np.float32)
        ref   = torch.nn.functional.layer_norm(
                    torch.from_numpy(x), [n],
                    weight=torch.from_numpy(gamma),
                    bias=torch.from_numpy(beta)).numpy()
        out   = simd_kernels.layer_norm(x, gamma=gamma, beta=beta)
        max_err = float(np.max(np.abs(out - ref) / (np.abs(ref) + 1e-7)))
        assert max_err < self.LN_TOL, f"vs PyTorch LayerNorm: max_rel_err={max_err:.2e}"


# ─── Alignment / build_info Tests ─────────────────────────────────────────────

class TestBuildInfo:
    def test_build_info_returns_dict(self):
        info = simd_kernels.build_info()
        assert isinstance(info, dict)
        assert "isa_compiled" in info or "ISA" in info

    def test_detected_isa_is_string(self):
        isa = simd_kernels.detected_isa()
        assert isinstance(isa, str)
        assert any(kw in isa for kw in ("avx", "scalar", "sse"))

    def test_is_aligned_returns_bool(self):
        x = np.zeros(1024, dtype=np.float32)
        result = simd_kernels.is_aligned(x)
        assert isinstance(result, bool)
