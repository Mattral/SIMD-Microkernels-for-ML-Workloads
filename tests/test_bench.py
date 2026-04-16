"""
tests/test_bench.py — Python-side Benchmark with Matplotlib Visualisation
 
Measures SIMD vs scalar performance for GEMM and GeLU across a range of
problem sizes, then renders two publication-quality plots:
 
  1. GEMM: Wall-clock time (ms) and GFLOPS vs matrix size N
  2. GeLU: Throughput (GElements/s) and speedup vs input length N
 
Usage:
    python tests/test_bench.py              # run + show + save plots
    python tests/test_bench.py --no-plot    # run + print table only
    python tests/test_bench.py --quick      # small sizes only (CI smoke test)
    python tests/test_bench.py --save-dir results/
 
Requirements:
    pip install numpy matplotlib tabulate
    # simd_kernels must be installed:  pip install -e .
"""
 
import argparse
import time
import sys
from pathlib import Path
 
import numpy as np
 
# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    import simd_kernels
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    print("WARNING: simd_kernels not found. Only NumPy reference timings will run.")
 
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
 
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
 
 
# ─── Timer utility ────────────────────────────────────────────────────────────
 
def timeit(fn, warmup: int = 3, reps: int = 10) -> tuple[float, float]:
    """
    Returns (min_seconds, median_seconds) over `reps` timed runs,
    after `warmup` untimed warm-up passes.
    Uses time.perf_counter (ns resolution on Linux via clock_gettime).
    """
    for _ in range(warmup):
        fn()
 
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
 
    times.sort()
    return times[0], times[len(times) // 2]
 
 
def gelu_numpy_ref(x: np.ndarray) -> np.ndarray:
    """Reference GeLU (tanh approximation) in NumPy."""
    c = np.sqrt(2.0 / np.pi).astype(np.float32)
    return (0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))).astype(np.float32)
 
 
# ─── GEMM benchmark ───────────────────────────────────────────────────────────
 
def bench_gemm(sizes, reps: int = 10) -> list[dict]:
    rng = np.random.default_rng(42)
    results = []
 
    for N in sizes:
        M, K = N, N
        flops = 2 * M * N * K  # multiply-add counted as 2 ops
 
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
 
        # NumPy reference (uses OpenBLAS/MKL internally)
        t_np_min, t_np_med = timeit(lambda: np.dot(A, B), reps=reps)
        gflops_np = flops / t_np_min / 1e9
 
        row = {
            "N": N,
            "flops": flops,
            "numpy_ms": t_np_min * 1e3,
            "numpy_gflops": gflops_np,
            "simd_ms": None,
            "simd_gflops": None,
            "speedup": None,
        }
 
        if HAS_SIMD:
            t_simd_min, _ = timeit(
                lambda: simd_kernels.sgemm(A, B, C, alpha=1.0, beta=0.0),
                reps=reps
            )
            gflops_simd = flops / t_simd_min / 1e9
            row["simd_ms"] = t_simd_min * 1e3
            row["simd_gflops"] = gflops_simd
            row["speedup"] = t_np_min / t_simd_min  # relative to numpy
 
        results.append(row)
        _print_gemm_row(row)
 
    return results
 
 
def _print_gemm_row(r: dict):
    simd_str = (
        f"{r['simd_ms']:7.3f} ms  {r['simd_gflops']:6.1f} GFLOPS  {r['speedup']:5.2f}x"
        if r["simd_ms"] is not None else "  (simd_kernels not installed)"
    )
    print(f"  GEMM {r['N']:4d}×{r['N']:4d}  "
          f"numpy: {r['numpy_ms']:7.3f} ms  {r['numpy_gflops']:6.1f} GFLOPS  |  "
          f"simd: {simd_str}")
 
 
# ─── GeLU benchmark ───────────────────────────────────────────────────────────
 
def bench_gelu(sizes, reps: int = 10) -> list[dict]:
    rng = np.random.default_rng(99)
    results = []
 
    for n in sizes:
        x = rng.uniform(-3.0, 3.0, size=n).astype(np.float32)
 
        t_np_min, _ = timeit(lambda: gelu_numpy_ref(x), reps=reps)
        gelems_np = n / t_np_min / 1e9  # billion elements/s
 
        row = {
            "n": n,
            "numpy_ms": t_np_min * 1e3,
            "numpy_gelems": gelems_np,
            "simd_ms": None,
            "simd_gelems": None,
            "speedup": None,
        }
 
        if HAS_SIMD:
            out = np.empty_like(x)
            t_simd_min, _ = timeit(
                lambda: simd_kernels.gelu_inplace(x.copy()),
                reps=reps
            )
            gelems_simd = n / t_simd_min / 1e9
            row["simd_ms"] = t_simd_min * 1e3
            row["simd_gelems"] = gelems_simd
            row["speedup"] = t_np_min / t_simd_min
 
        results.append(row)
        _print_gelu_row(row)
 
    return results
 
 
def _print_gelu_row(r: dict):
    simd_str = (
        f"{r['simd_ms']:8.4f} ms  {r['simd_gelems']:5.2f} GElem/s  {r['speedup']:5.2f}x"
        if r["simd_ms"] is not None else "  (simd_kernels not installed)"
    )
    print(f"  GeLU n={r['n']:8d}  "
          f"numpy: {r['numpy_ms']:8.4f} ms  {r['numpy_gelems']:5.2f} GElem/s  |  "
          f"simd: {simd_str}")
 
 
# ─── Table printer ────────────────────────────────────────────────────────────
 
def print_gemm_table(results: list[dict]):
    rows = []
    for r in results:
        rows.append([
            f"{r['N']}×{r['N']}",
            f"{r['numpy_ms']:.3f}",
            f"{r['numpy_gflops']:.1f}",
            f"{r['simd_ms']:.3f}"    if r["simd_ms"]     is not None else "—",
            f"{r['simd_gflops']:.1f}" if r["simd_gflops"] is not None else "—",
            f"{r['speedup']:.2f}x"   if r["speedup"]     is not None else "—",
        ])
    headers = ["Size", "NumPy ms", "NumPy GFLOPS", "SIMD ms", "SIMD GFLOPS", "Speedup"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("  " + " | ".join(f"{h:>12}" for h in headers))
        for row in rows:
            print("  " + " | ".join(f"{v:>12}" for v in row))
 
 
def print_gelu_table(results: list[dict]):
    rows = []
    for r in results:
        rows.append([
            f"{r['n']:,}",
            f"{r['numpy_ms']:.4f}",
            f"{r['numpy_gelems']:.2f}",
            f"{r['simd_ms']:.4f}"    if r["simd_ms"]    is not None else "—",
            f"{r['simd_gelems']:.2f}" if r["simd_gelems"] is not None else "—",
            f"{r['speedup']:.2f}x"   if r["speedup"]    is not None else "—",
        ])
    headers = ["N Elements", "NumPy ms", "NumPy GEl/s", "SIMD ms", "SIMD GEl/s", "Speedup"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("  " + " | ".join(f"{h:>12}" for h in headers))
        for row in rows:
            print("  " + " | ".join(f"{v:>12}" for v in row))
 
 
# ─── Plots ────────────────────────────────────────────────────────────────────
 
def plot_gemm(results: list[dict], save_dir: Path | None = None):
    if not HAS_MPL:
        print("matplotlib not installed — skipping GEMM plot.")
        return
 
    sizes      = [r["N"] for r in results]
    np_gflops  = [r["numpy_gflops"] for r in results]
    simd_gflops = [r["simd_gflops"]  for r in results if r["simd_gflops"] is not None]
    simd_sizes  = [r["N"]            for r in results if r["simd_gflops"] is not None]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GEMM Performance: SIMD AVX2 vs NumPy (OpenBLAS)",
                 fontsize=13, fontweight="bold")
 
    # Left: GFLOPS
    ax = axes[0]
    ax.plot(sizes, np_gflops,   "o--", color="#5B8DB8", linewidth=2,
            markersize=7, label="NumPy / OpenBLAS")
    if simd_sizes:
        ax.plot(simd_sizes, simd_gflops, "s-", color="#E07B39", linewidth=2,
                markersize=7, label="IntrinsicML SIMD")
    ax.set_xlabel("Matrix Size N (N×N×N)", fontsize=11)
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_title("Arithmetic Throughput", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
 
    # Right: Speedup bar chart
    if simd_sizes:
        ax2 = axes[1]
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]
        bars = ax2.bar([str(n) for n in simd_sizes], speedups,
                       color="#E07B39", edgecolor="black", linewidth=0.6, alpha=0.85)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="NumPy baseline")
        ax2.set_xlabel("Matrix Size N", fontsize=11)
        ax2.set_ylabel("Speedup vs NumPy", fontsize=11)
        ax2.set_title("Relative Speedup (higher = better)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, sp in zip(bars, speedups):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{sp:.2f}×", ha="center", va="bottom", fontsize=9)
 
    plt.tight_layout()
    if save_dir:
        out = save_dir / "bench_gemm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.show()
 
 
def plot_gelu(results: list[dict], save_dir: Path | None = None):
    if not HAS_MPL:
        print("matplotlib not installed — skipping GeLU plot.")
        return
 
    ns          = [r["n"]            for r in results]
    np_gelems   = [r["numpy_gelems"] for r in results]
    simd_gelems = [r["simd_gelems"]  for r in results if r["simd_gelems"] is not None]
    simd_ns     = [r["n"]            for r in results if r["simd_gelems"] is not None]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GeLU Performance: SIMD AVX2 vs NumPy",
                 fontsize=13, fontweight="bold")
 
    # Left: Throughput
    ax = axes[0]
    ax.plot(ns, np_gelems, "o--", color="#5B8DB8", linewidth=2,
            markersize=7, label="NumPy reference")
    if simd_ns:
        ax.plot(simd_ns, simd_gelems, "s-", color="#E07B39", linewidth=2,
                markersize=7, label="IntrinsicML SIMD")
    ax.set_xlabel("Input Elements (N)", fontsize=11)
    ax.set_ylabel("GElements / second", fontsize=11)
    ax.set_title("GeLU Activation Throughput", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
 
    # Right: Speedup line
    if simd_ns:
        ax2 = axes[1]
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]
        ax2.plot(simd_ns, speedups, "D-", color="#6ABF69", linewidth=2,
                 markersize=8)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
        ax2.fill_between(simd_ns, 1.0, speedups, alpha=0.15, color="#6ABF69")
        ax2.set_xlabel("Input Elements (N)", fontsize=11)
        ax2.set_ylabel("Speedup vs NumPy", fontsize=11)
        ax2.set_title("Relative Speedup (higher = better)", fontsize=11)
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.grid(True, alpha=0.3)
        for x, y in zip(simd_ns, speedups):
            ax2.annotate(f"{y:.2f}×", (x, y),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
 
    plt.tight_layout()
    if save_dir:
        out = save_dir / "bench_gelu.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.show()
 
 
# ─── Entry point ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="IntrinsicML benchmark suite")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip matplotlib rendering (print tables only)")
    parser.add_argument("--quick",    action="store_true",
                        help="Run small sizes only (CI smoke test)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory to save PNG plots into")
    parser.add_argument("--reps",     type=int, default=10,
                        help="Timing repetitions per measurement (default: 10)")
    args = parser.parse_args()
 
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
 
    # Problem sizes
    if args.quick:
        gemm_sizes = [32, 64, 128]
        gelu_sizes = [1024, 8192, 65536]
    else:
        gemm_sizes = [32, 64, 128, 256, 512, 1024]
        gelu_sizes = [1024, 4096, 16384, 65536, 262144, 1 << 20]
 
    show_plots = HAS_MPL and not args.no_plot
 
    # ── GEMM ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  GEMM Benchmark  (C = A × B, FP32, square matrices)")
    print("═" * 70)
    gemm_results = bench_gemm(gemm_sizes, reps=args.reps)
    print()
    print_gemm_table(gemm_results)
 
    # ── GeLU ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  GeLU Benchmark  (element-wise, FP32, tanh approximation)")
    print("═" * 70)
    gelu_results = bench_gelu(gelu_sizes, reps=args.reps)
    print()
    print_gelu_table(gelu_results)
 
    # ── Plots ─────────────────────────────────────────────────────────────────
    if show_plots:
        print("\nRendering plots...")
        plot_gemm(gemm_results, save_dir=args.save_dir)
        plot_gelu(gelu_results, save_dir=args.save_dir)
    elif not HAS_MPL:
        print("\nInstall matplotlib for plots:  pip install matplotlib")
 
    # ── Exit code for CI ──────────────────────────────────────────────────────
    # Fail if any SIMD result is slower than NumPy (regression guard)
    if HAS_SIMD:
        regressions = [r for r in gemm_results + gelu_results
                       if r.get("speedup") is not None and r["speedup"] < 0.5]
        if regressions:
            print("\nPERFORMANCE REGRESSION: SIMD kernel is >2× slower than NumPy.")
            for r in regressions:
                print(f"  {r}")
            sys.exit(1)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
