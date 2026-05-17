# Monte Carlo simulation using Black-Scholes for stock price in Python

Published: 2024-03-29  
Medium: [Monte Carlo simulation using Black-Scholes for stock price in Python](https://medium.com/@kyle-t-jones/monte-carlo-simulation-using-black-scholes-for-stock-price-in-python-808574935473)

Geometric Brownian motion Monte Carlo paths for forecasting terminal price distributions. Companion code for the article (`article.md`).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run monte-carlo-run
```

Outputs:

| Path | Contents |
|------|----------|
| `outputs/figures/` | Price history, simulated paths, terminal histogram |
| `outputs/results.json` | Summary stats (mean, std, P(price > spot)) |

## Project layout

```
config.yaml              # data paths, simulation horizon, output settings
config.local.yaml.example
pyproject.toml / uv.lock
src/monte_carlo_bs/      # simulation, plotting, CLI
scripts/fetch_prices.py  # refresh data/aoil.csv via yfinance
notebooks/               # original exploratory notebook
data/aoil.csv            # bundled historical prices
outputs/figures/         # generated plots (gitignored except .gitkeep)
tests/
article.md
```

## Configuration

Edit `config.yaml`:

- `data.source` — `csv` (default) or `yfinance`
- `simulation.forecast_end` — last date in the forecast window
- `simulation.iterations` — number of paths (default 1000)
- `simulation.use_weekdays_only` — `true` matches the original notebook; `false` uses business days (article style)

Machine-specific overrides: copy `config.local.yaml.example` to `config.local.yaml` (gitignored).

## Data

See [data/README.md](data/README.md). To refresh the CSV:

```bash
uv sync --extra fetch
uv run python scripts/fetch_prices.py
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts
```

CI runs ruff and pytest on push/PR (see `.github/workflows/ci.yml`).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
