# Monte Carlo simulation using Black-Scholes for stock price in Python

Published: 2024-03-29  
Medium: [Monte Carlo simulation using Black-Scholes for stock price in Python](https://medium.com/@kyle-t-jones/monte-carlo-simulation-using-black-scholes-for-stock-price-in-python-808574935473)

Geometric Brownian motion Monte Carlo paths for forecasting terminal price distributions. Companion code for the article (`article.md`).

## Business context

This project continues another project I did looking at the price of Telsa stock. Please note, this is not financial advice, just a fun...

This project continues another project I did looking at the price of Telsa stock. Please note, this is not financial advice, just a fun project.

I wanted to combine this approach with the [Black-Scholes algo](https://www.investopedia.com/terms/b/blackscholes.asp) and then use Monte Carlo simulations to predict the value of the Tesla stock in ~90 days.

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
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
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

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — GBM terminal price simulation. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p monte_carlo_simulation_using_black_scholes_for_stock_price_in_python_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).