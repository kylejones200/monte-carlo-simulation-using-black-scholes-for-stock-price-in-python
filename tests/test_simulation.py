from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from monte_carlo_bs.simulation import business_forecast_dates, monte_carlo_gbm


@pytest.fixture
def prices() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.Series(values, index=idx, name="price")


def test_business_forecast_dates(prices: pd.Series) -> None:
    end = datetime(2020, 6, 1)
    dates = business_forecast_dates(prices.index.max(), end)
    assert len(dates) > 0
    assert dates[0] > prices.index.max()


def test_monte_carlo_shapes(prices: pd.Series) -> None:
    end = datetime(2020, 7, 1)
    forecast_df, final_prices, forecast_dates = monte_carlo_gbm(
        prices, end, iterations=50, seed=1
    )
    assert len(forecast_dates) == len(forecast_df)
    assert forecast_df.shape[1] == 50
    assert final_prices.shape == (50,)


def test_monte_carlo_reproducible(prices: pd.Series) -> None:
    end = datetime(2020, 7, 1)
    _, a, _ = monte_carlo_gbm(prices, end, iterations=10, seed=99)
    _, b, _ = monte_carlo_gbm(prices, end, iterations=10, seed=99)
    np.testing.assert_array_equal(a, b)
