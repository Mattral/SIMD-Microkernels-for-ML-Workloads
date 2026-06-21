"""
tests/conftest.py — pytest configuration for the IntrinsicML test suite.

Adds the project root and build directories to sys.path so that
`import simd_kernels` resolves to the compiled extension without requiring
a system-wide `pip install`.  The search order is:

  1. build_ci/   (CMake out-of-source build used locally and in CI)
  2. build/      (alternative build directory name)
  3. Project root (if .so was copied there manually)

This is intentionally a fallback mechanism.  The canonical way to import
`simd_kernels` in CI is `pip install -e .` (via scikit-build-core), which
puts the .so on the Python path permanently.  conftest.py exists so that
`pytest tests/` works immediately after `cmake --build build_ci` without
any install step.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Repository root is one level above this file (tests/conftest.py → repo root)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Candidate build directories, in preference order
_SEARCH_DIRS = [
    REPO_ROOT / "build_ci",
    REPO_ROOT / "build",
    REPO_ROOT,          # .so copied to root (common manual step)
]


def _try_add_path(directory: Path) -> bool:
    """Return True if directory contains a simd_kernels .so and was added to sys.path."""
    so_files = list(directory.glob("simd_kernels*.so")) + \
               list(directory.glob("simd_kernels*.pyd"))   # Windows
    if so_files:
        str_dir = str(directory)
        if str_dir not in sys.path:
            sys.path.insert(0, str_dir)
        return True
    return False


for _d in _SEARCH_DIRS:
    if _try_add_path(_d):
        break
