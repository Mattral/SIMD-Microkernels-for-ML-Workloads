#!/usr/bin/env python3
"""
tests/smoke_test.py — Minimal post-install verification script.

Run after `pip install -e .` to confirm the compiled extension loads and the
core kernels produce correctly-shaped, finite output. This exists as a
standalone script (rather than an inline CI heredoc) because YAML multiline
blocks are fragile: indentation, quoting, and shell-vs-Python escaping easily
break heredocs across different runner shells. A real .py file is testable
locally with `python3 tests/smoke_test.py` and behaves identically in CI.
"""
import sys

import numpy as np


def main() -> int:
    import simd_kernels

    print("simd_kernels module file:", simd_kernels.__file__)
    print("build_info():", simd_kernels.build_info())
    print("detected_isa():", simd_kernels.detected_isa())

    rng = np.random.default_rng(0)

    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 32)).astype(np.float32)
    C = simd_kernels.sgemm(A, B)
    assert C.shape == (32, 32), f"sgemm: bad shape {C.shape}"
    assert np.all(np.isfinite(C)), "sgemm: non-finite output"

    x = rng.standard_normal(1024).astype(np.float32)

    y = simd_kernels.gelu(x)
    assert y.shape == x.shape, f"gelu: bad shape {y.shape}"
    assert np.all(np.isfinite(y)), "gelu: non-finite output"

    y = simd_kernels.relu(x)
    assert y.shape == x.shape, f"relu: bad shape {y.shape}"

    y = simd_kernels.silu(x)
    assert y.shape == x.shape, f"silu: bad shape {y.shape}"

    y = simd_kernels.softmax(x)
    assert y.shape == x.shape, f"softmax: bad shape {y.shape}"
    assert abs(float(y.sum()) - 1.0) < 1e-4, "softmax: rows do not sum to 1"

    y = simd_kernels.layer_norm(x)
    assert y.shape == x.shape, f"layer_norm: bad shape {y.shape}"

    gemm = simd_kernels.GEMMConfig(alpha=1.0, beta=0.0)
    C2 = gemm(A, B)
    assert np.allclose(C, C2, rtol=1e-4), "GEMMConfig output diverges from sgemm"

    print("Smoke test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
