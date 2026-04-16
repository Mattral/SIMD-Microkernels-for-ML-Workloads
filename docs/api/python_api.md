# Python API

The Python extension exposes the core math kernels through a clean API.

## Available functions

### `sgemm(A, B, C=None)`

Performs single-precision matrix multiplication in the form `A @ B`.

Parameters:

* `A` — 2D NumPy-like array of shape `(M, K)` and dtype `float32`
* `B` — 2D NumPy-like array of shape `(K, N)` and dtype `float32`
* `C` — optional output buffer of shape `(M, N)` and dtype `float32`

Returns:

* `C` — a NumPy array of shape `(M, N)` containing the result

Example:

```python
import numpy as np
import simd_kernels

A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
C = simd_kernels.sgemm(A, B)
```

If `C` is provided, the result is written into the provided buffer and returned.

### `gelu(X)`

Applies the GeLU activation function to the input array.

Parameters:

* `X` — 1D or 2D NumPy-like array of dtype `float32`

Returns:

* `Y` — a NumPy array with the same shape as `X`

Example:

```python
X = np.linspace(-3, 3, 128, dtype=np.float32)
y = simd_kernels.gelu(X)
```

### `build_info()`

Returns build metadata as a Python dictionary.

Example:

```python
info = simd_kernels.build_info()
print(info)
```

Sample keys:

* `compiler` — compiler identifier
* `build_type` — CMake build type
* `simd` — enabled SIMD target description
* `openmp` — whether OpenMP support was enabled

## Type stubs

This project provides `simd_kernels.pyi` for editors and type checkers.
The stubs declare the expected function signatures and support better integration with Python development tooling.
