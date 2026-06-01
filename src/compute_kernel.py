"""GBM terminal price simulation (numpy)."""

from __future__ import annotations

import numpy as np


def gbm_terminal_prices(
    s0: float,
    log_drift: float,
    log_vol: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_paths, dtype=float)
    for p in range(n_paths):
        price = float(s0)
        for _ in range(n_steps):
            z = rng.standard_normal()
            price *= np.exp(log_drift + log_vol * z)
        out[p] = price
    return out
