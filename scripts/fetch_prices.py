#!/usr/bin/env python3
"""Download historical prices and write data/aoil.csv (notebook-compatible columns)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from monte_carlo_bs.paths import DEFAULT_PRICE_CSV, PROJECT_ROOT


def fetch_yfinance(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    import yfinance as yf

    kwargs: dict = {"progress": False, "auto_adjust": True}
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    raw = yf.download(ticker, **kwargs)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker!r}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.index),
            "adjClose": close.astype(float).values,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch prices into data/aoil.csv")
    parser.add_argument(
        "--ticker",
        default="USO",
        help="yfinance symbol (USO ≈ oil ETF; original notebook used AOIL via FMP)",
    )
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2020-03-01")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRICE_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()
    df = fetch_yfinance(args.ticker, start=args.start, end=args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
