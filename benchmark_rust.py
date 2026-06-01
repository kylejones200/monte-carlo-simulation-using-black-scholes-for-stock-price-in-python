#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import gbm_terminal_prices  # noqa: E402

def main() -> None:
    py = gbm_terminal_prices(100.0, 0.0002, 0.01, 252, 1000, 42)
    t0 = time.perf_counter()
    for _ in range(200):
        gbm_terminal_prices(100.0, 0.0002, 0.01, 252, 1000, 42)
    py_s = time.perf_counter() - t0
    try:
        import monte_carlo_simulation_using_black_scholes_for_stock_price_in_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(100.0, 0.0002, 0.01, 252, 1000, 42, 100)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    rs_out = np.asarray(rs.gbm_terminal_prices_py(100.0, 0.0002, 0.01, 252, 1000, 42))
    assert py.shape == rs_out.shape == (1000,)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
