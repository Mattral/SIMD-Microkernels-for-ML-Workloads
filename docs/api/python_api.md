# Python API Reference

**Module**: `simd_kernels`  
**Import**: `import simd_kernels`  
**Type stubs**: `simd_kernels.pyi` (IDE autocompletion)

All kernels operate on **float32 NumPy arrays**, release the **GIL** during computation (threading-safe), and require **C-contiguous** memory layout. Call `numpy.ascontiguousarray(arr)` if needed.

---

## Linear Algebra

### `sgemm(A, B, C=None, alpha=1.0, beta=0.0) → np.ndarray`

Compute `C = alpha * A @ B + beta * C` (single-precision GEMM).

| Parameter | Type | Description |
|---|---|---|
| `A` | `ndarray[float32]`, shape `[M, K]` | Left operand |
| `B` | `ndarray[float32]`, shape `[K, N]` | Right operand |
| `C` | `ndarray[float32]`, shape `[M, N]`, optional | Output (allocated if `None`) |
| `alpha` | `float` | Scale factor for `A @ B` (default `1.0`) |
| `beta` | `float` | Scale factor for existing `C` (`0.0` = overwrite) |

Returns the `[M, N]` result array (same object as `C` if provided).

```python
import numpy as np, simd_kernels

A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)

# Simple GEMM (allocates output)
C = simd_kernels.sgemm(A, B)

# In-place with scaling: C = 2*A@B + 0.5*C
simd_kernels.sgemm(A, B, C, alpha=2.0, beta=0.5)
```

**Implementation**: Goto/BLIS 5-loop structure with panel packing (8×8 register tile, AVX2 FMA). See `docs/DESIGN.md §2–3`.

### `GEMMConfig` — callable GEMM configuration object

```python
class GEMMConfig:
    def __init__(self, alpha=1.0, beta=0.0, isa="",
                 tile_m=128, tile_n=2048, tile_k=256, mr=8, nr=8): ...
    def __call__(self, A, B, C=None, alpha=None, beta=None) -> np.ndarray: ...
```

Stores default `alpha`, `beta`, and `isa` for repeated GEMM calls. Tile
parameters (`tile_m`, `tile_n`, `tile_k`, `mr`, `nr`) are accepted for API
completeness and are validated (must be positive), but do not currently
change dispatch — the compiled kernel uses fixed compile-time constants.
Auto-tuned tile selection is planned (`docs/ROADMAP.md §v0.9`).

```python
gemm = simd_kernels.GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
C = gemm(A, B)                      # uses stored alpha=2.0, beta=0.5
C = gemm(A, B, alpha=1.0, beta=0.0) # per-call override
print(gemm)                          # GEMMConfig(alpha=2.0, beta=0.5, isa='avx2')
```

`isa` must be one of `""`, `"avx2"`, `"avx512"`, `"scalar"` — an invalid
value raises `ValueError` at construction time. The `isa=` parameter is also
accepted directly by `sgemm()`:

```python
C = simd_kernels.sgemm(A, B, isa="avx2")   # validated, currently informational
```

**Why is `isa=` "currently informational"?** The runtime dispatcher
(`kernel_registry.hpp`) already selects the best available SIMD path via
CPUID at process startup — there is one compiled GEMM kernel per ISA tier,
not a runtime-selectable set. The `isa=` parameter exists today for forward
API compatibility: once the dual-accumulator AVX-512 kernel lands
(`docs/ROADMAP.md §v0.8`), this parameter will let callers explicitly select
between the AVX2 and AVX-512 paths on AVX-512-capable hardware.

---

## Activation Functions

### `gelu(x) → np.ndarray`

Out-of-place GeLU activation (tanh approximation). Input `x` is unchanged.

```python
x = np.random.randn(65536).astype(np.float32)
y = simd_kernels.gelu(x)   # y = GeLU(x), x unchanged
```

### `gelu_inplace(x) → None`

In-place GeLU: `x[:] = GeLU(x)`. No allocation. `x` must be writeable.

```python
simd_kernels.gelu_inplace(x)   # modifies x in place
```

Both functions use the tanh approximation from BERT/GPT:
```
GeLU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```
Max absolute error vs exact GeLU: < 5×10⁻⁴ (formula gap, not implementation error).  
Max absolute error vs `torch.nn.functional.gelu(x, approximate='tanh')`: < 5×10⁻⁷.

### `relu(x) → np.ndarray`

ReLU activation: `output[i] = max(0, x[i])`. Exact (no approximation).

```python
y = simd_kernels.relu(x)
```

### `silu(x) → np.ndarray`

SiLU/Swish activation: `output[i] = x[i] * sigmoid(x[i])`.  
Implemented as `x · 0.5·(1 + tanh(x/2))` using the Cody–Waite fast tanh.  
Max absolute error: < 2×10⁻⁷ vs float32 reference.

```python
y = simd_kernels.silu(x)
```

### `softmax(x, axis=-1) → np.ndarray`

Numerically stable softmax over the specified axis (default: last axis).  
Uses max-subtraction and double-precision accumulation to prevent overflow.

```python
logits = np.random.randn(32, 1000).astype(np.float32)
probs  = simd_kernels.softmax(logits)       # shape [32, 1000], rows sum to 1
```

Currently only the **last axis** (`axis=-1`) is supported. For other axes, transpose first.

### `layer_norm(x, gamma=None, beta=None, eps=1e-5) → np.ndarray`

Layer normalization over the last axis of `x`:

```
output = (x - mean) / sqrt(var + eps) * gamma + beta
```

`gamma` and `beta` are optional learnable affine parameters with shape matching the last dimension of `x`.

```python
x     = np.random.randn(16, 768).astype(np.float32)
gamma = np.ones(768,  dtype=np.float32)   # scale
beta  = np.zeros(768, dtype=np.float32)   # shift

y = simd_kernels.layer_norm(x, gamma=gamma, beta=beta)
# y has zero mean and unit std per row (before affine)
```

**Implementation**: 3-pass AVX2 (mean → variance → normalize) with double-precision horizontal accumulation to prevent catastrophic cancellation.

---

## Threading

### `set_num_threads(n: int) → None`

Set the number of OpenMP threads used by `sgemm`. No-op when built without `-DSIMD_ML_OPENMP`.

### `get_num_threads() → int`

Return the current OpenMP thread count (always 1 without OpenMP).

---

## Diagnostics

### `build_info() → dict`

Return build metadata as a Python dictionary.

```python
info = simd_kernels.build_info()
# Example output:
# {
#   'isa_compiled': 'AVX2+FMA3',
#   'fma': True,
#   'alignment': '64-byte (posix_memalign / _aligned_malloc)',
#   'build_timestamp': 'Jun 20 2026 14:22:11',
#   'runtime_isa': 'avx2'
# }
```

### `detected_isa() → str`

Return the runtime-detected ISA string: `'avx512'`, `'avx2'`, `'sse42'`, or `'scalar'`.

### `is_aligned(arr: np.ndarray) → bool`

Return `True` if `arr.data` is 64-byte aligned. Useful for verifying that NumPy arrays created with `np.empty` or `np.zeros` meet the alignment contract.

```python
x = np.empty(1024, dtype=np.float32)
simd_kernels.is_aligned(x)   # True on most NumPy allocations ≥ 64 elements
```

---

## Error Handling

All functions raise `RuntimeError` with a descriptive message for invalid inputs:

```python
# Wrong dtype
simd_kernels.gelu(np.array([1.0, 2.0], dtype=np.float64))
# RuntimeError: x must be float32 (dtype=np.float32). Call arr.astype(np.float32)...

# Non-contiguous array
simd_kernels.relu(arr[::2])
# RuntimeError: x must be C-contiguous. Call numpy.ascontiguousarray(arr)...

# Shape mismatch
simd_kernels.sgemm(np.zeros((4, 8), dtype=np.float32),
                   np.zeros((3, 4), dtype=np.float32))
# RuntimeError: Shape mismatch: A is [4×8] but B is [3×4]
```

---

## Type Stubs

The file `simd_kernels.pyi` provides full type annotations for IDE support (Pylance, Pyright, mypy). After `pip install -e .`, type checkers will resolve `simd_kernels` automatically.
