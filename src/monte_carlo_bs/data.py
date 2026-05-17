from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from monte_carlo_bs.paths import resolve_project_path

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("adjClose", "Adj Close", "close", "Close")


def _pick_price_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for name in PRICE_COLUMNS:
        if name in df.columns:
            return name
    raise ValueError(
        f"No price column found in {list(df.columns)}; expected one of {PRICE_COLUMNS}"
    )


def load_prices(
    config_path: Path | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load historical prices as a DataFrame indexed by date with a ``price`` column.

    Returns ``(df, ticker)``.
    """
    if cfg is None:
        from monte_carlo_bs.config import load_config

        path = Path(config_path) if config_path else None
        cfg = load_config(path)

    data_cfg = cfg.get("data") or {}
    source = str(data_cfg.get("source", "csv")).lower()
    ticker = str(data_cfg.get("display_ticker") or data_cfg.get("ticker", "USO"))

    if source == "csv":
        csv_rel = data_cfg.get("csv_path", "data/aoil.csv")
        csv_path = resolve_project_path(csv_rel)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Price CSV not found at {csv_path}. "
                "Run `uv run python scripts/fetch_prices.py` or see data/README.md."
            )
        df = pd.read_csv(csv_path)
        date_col = data_cfg.get("date_column", "date")
        if date_col not in df.columns:
            raise ValueError(f"Column {date_col!r} missing from {csv_path}")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        price_col = _pick_price_column(df, data_cfg.get("price_column"))
        out = pd.DataFrame({"price": df[price_col].astype(float)})
        return out, ticker

    if source == "yfinance":
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for data.source=yfinance. "
                "Install with: uv sync --extra fetch"
            ) from exc

        period = data_cfg.get("period", "5y")
        start = data_cfg.get("start")
        end = data_cfg.get("end")
        kwargs: dict[str, Any] = {"progress": False, "auto_adjust": True}
        if start:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        raw = yf.download(ticker, **kwargs)
        if raw.empty:
            raise ValueError(f"No rows returned from yfinance for {ticker!r}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in raw.columns]

        price_col = _pick_price_column(raw, data_cfg.get("price_column"))
        out = pd.DataFrame({"price": raw[price_col].astype(float)})
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        return out, ticker

    raise ValueError(f"Unknown data.source: {source!r} (use 'csv' or 'yfinance')")


def parse_forecast_end(cfg: dict[str, Any], last_date: pd.Timestamp) -> datetime:
    sim = cfg.get("simulation") or {}
    end = sim.get("forecast_end")
    if end is None:
        raise ValueError("simulation.forecast_end is required in config.yaml")
    pred_end = pd.Timestamp(end).to_pydatetime()
    if pred_end <= last_date.to_pydatetime():
        raise ValueError(
            f"forecast_end {pred_end.date()} must be after last price date {last_date.date()}"
        )
    return pred_end
