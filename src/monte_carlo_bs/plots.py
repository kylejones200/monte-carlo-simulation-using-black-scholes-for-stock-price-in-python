from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monte_carlo_bs.paths import resolve_project_path

logger = logging.getLogger(__name__)


def _figures_dir(cfg: dict[str, Any]) -> Path:
    out = cfg.get("output") or {}
    rel = out.get("figures_dir", "outputs/figures")
    path = resolve_project_path(rel)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figure_settings(cfg: dict[str, Any]) -> tuple[int, str]:
    out = cfg.get("output") or {}
    return int(out.get("figure_dpi", 120)), str(out.get("figure_format", "png"))


def plot_price_history(
    df: pd.DataFrame,
    ticker: str,
    cfg: dict[str, Any],
) -> Path:
    """Historical price series."""
    figures_dir = _figures_dir(cfg)
    dpi, fmt = _figure_settings(cfg)
    out_cfg = cfg.get("output") or {}
    filename = out_cfg.get("history_chart", "price_history.png")
    out_path = figures_dir / filename
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["price"], color="#333333", linewidth=0.8)
    ax.set_title(f"Price of {ticker} from {df.index.min().date()} to {df.index.max().date()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_simulated_paths(
    forecast_df: pd.DataFrame,
    ticker: str,
    iterations: int,
    cfg: dict[str, Any],
) -> Path:
    """Spaghetti plot of simulated future paths."""
    figures_dir = _figures_dir(cfg)
    dpi, fmt = _figure_settings(cfg)
    out_cfg = cfg.get("output") or {}
    filename = out_cfg.get("paths_chart", "simulated_paths.png")
    out_path = figures_dir / filename
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in forecast_df.columns:
        ax.plot(
            forecast_df.index,
            forecast_df[col],
            color="#666666",
            alpha=0.15,
            linewidth=0.5,
        )
    ax.set_title(f"{iterations} simulated future paths ({ticker})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_terminal_histogram(
    final_prices: np.ndarray,
    spot: float,
    ticker: str,
    pred_end_date: datetime,
    cfg: dict[str, Any],
) -> Path:
    """Histogram of terminal prices with normal overlay (notebook style)."""
    figures_dir = _figures_dir(cfg)
    dpi, fmt = _figure_settings(cfg)
    out_cfg = cfg.get("output") or {}
    filename = out_cfg.get("histogram_chart", "terminal_distribution.png")
    out_path = figures_dir / filename
    x = np.asarray(final_prices, dtype=float)
    sigma = float(np.std(x))
    mu = float(np.mean(x))
    num_bins = int((cfg.get("output") or {}).get("histogram_bins", 100))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, num_bins, density=True, alpha=0.75, color="#4a6fa5")
    bins = np.linspace(x.min(), x.max(), num_bins + 1)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma) ** 2)
    ax.plot(bins, y, "--", color="#333333")
    ax.axvline(mu, color="r", label=f"mean ({mu:.2f})")
    ax.axvline(mu + sigma * 1.96, color="g", ls="--", label="±1.96σ")
    ax.axvline(mu - sigma * 1.96, color="g", ls="--")
    ax.axvline(spot, color="#c44e52", label=f"spot ({spot:.2f})")
    ax.set_xlabel(f"Predicted price on {pred_end_date.date()}")
    ax.set_ylabel("Probability density")
    ax.set_title(rf"Terminal distribution ({ticker}): $\mu={mu:.2f}$, $\sigma={sigma:.2f}$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def run_plots(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    final_prices: np.ndarray,
    ticker: str,
    pred_end_date: datetime,
    iterations: int,
    cfg: dict[str, Any],
) -> dict[str, Path]:
    spot = float(df["price"].iloc[-1])
    return {
        "history": plot_price_history(df, ticker, cfg),
        "paths": plot_simulated_paths(forecast_df, ticker, iterations, cfg),
        "histogram": plot_terminal_histogram(final_prices, spot, ticker, pred_end_date, cfg),
    }
