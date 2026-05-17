# Data

## Bundled file

`aoil.csv` — daily prices (columns: `date`, `adjClose`). Filename kept for notebook compatibility.

The committed snapshot is **USO** (United States Oil Fund), 2017–2020 — a public proxy for the Aberdeen Oil (AOIL) index from the original notebook, which requires a [Financial Modeling Prep](https://financialmodelingprep.com/) API key. `config.yaml` sets `data.ticker: USO` so plots and results match the file.

## Refresh or use another symbol

```bash
uv sync --extra fetch
uv run python scripts/fetch_prices.py --ticker USO --start 2017-01-01 --end 2020-03-01
```

## Live download in the pipeline

Set `data.source: yfinance` in `config.yaml` (see commented example in `config.local.yaml.example`).
