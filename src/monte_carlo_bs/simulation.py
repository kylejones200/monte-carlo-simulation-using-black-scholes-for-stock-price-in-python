from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def business_forecast_dates(
    last_date: pd.Timestamp,
    pred_end_date: datetime,
) -> pd.DatetimeIndex:
    """Business days from the day after ``last_date`` through ``pred_end_date``."""
    start = last_date + pd.Timedelta(days=1)
    end = pd.Timestamp(pred_end_date)
    dates = pd.date_range(start=start, end=end, freq="B")
    if dates.empty:
        raise ValueError(f"No business days between {last_date.date()} and {pred_end_date.date()}")
    return dates


def monte_carlo_gbm(
    prices: pd.Series,
    pred_end_date: datetime,
    *,
    iterations: int = 1000,
    seed: int | None = None,
    use_weekdays_only: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """
    Simulate future price paths with geometric Brownian motion (log-normal returns).
    Parameters
    ----------
    prices
        Historical adjusted close (or similar), indexed by date.
    pred_end_date
        Last calendar day to include in the forecast horizon.
    iterations
        Number of independent simulation paths.
    seed
        Optional RNG seed for reproducibility.
    use_weekdays_only
        If True, use the notebook-style weekday filter (Mon–Fri with NaN padding).
        If False (default), use business-day ``date_range`` (article style).

    Returns
    -------
    forecast_df
        Shape ``(intervals, iterations)`` with forecast dates as index.
    final_prices
        Terminal simulated prices, shape ``(iterations,)``.
    forecast_dates
        Index used for the simulation steps.
    """
    if seed is not None:
        np.random.seed(seed)

    last_date = prices.index.max()
    if use_weekdays_only:
        forecast_dates = pd.DatetimeIndex(
            [
                d if d.isoweekday() in range(1, 6) else pd.NaT
                for d in pd.date_range(last_date, pred_end_date)
            ]
        ).dropna()
        if len(forecast_dates) == 0:
            raise ValueError("No weekday forecast dates in range")
    else:
        forecast_dates = business_forecast_dates(last_date, pred_end_date)

    intervals = len(forecast_dates)
    log_returns = np.log(1 + prices.pct_change().dropna())
    u = float(log_returns.mean())
    var = float(log_returns.var())
    drift = u - 0.5 * var
    stdev = float(log_returns.std())
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(intervals, iterations)))
    s0 = float(prices.iloc[-1])
    price_list = np.zeros((intervals, iterations))
    price_list[0] = s0
    for t in range(1, intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    forecast_df = pd.DataFrame(price_list, index=forecast_dates)
    final_prices = forecast_df.iloc[-1].to_numpy()
    return forecast_df, final_prices, forecast_dates


def summarize_terminal(final_prices: np.ndarray) -> dict[str, Any]:
    """Summary statistics for terminal simulated prices."""
    x = np.asarray(final_prices, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p_above_spot": None,  # filled by caller when spot known
    }
