from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from monte_carlo_bs import __version__
from monte_carlo_bs.config import configure_logging, load_config
from monte_carlo_bs.data import load_prices, parse_forecast_end
from monte_carlo_bs.paths import (
    DEFAULT_CONFIG_PATH,
    path_relative_to_project,
    resolve_project_path,
)
from monte_carlo_bs.plots import run_plots
from monte_carlo_bs.simulation import monte_carlo_gbm, summarize_terminal

logger = logging.getLogger(__name__)


def run(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(path)
    configure_logging(cfg)
    df, ticker = load_prices(path, cfg)
    pred_end = parse_forecast_end(cfg, df.index.max())
    sim = cfg.get("simulation") or {}
    iterations = int(sim.get("iterations", 1000))
    seed = sim.get("seed")
    use_weekdays = bool(sim.get("use_weekdays_only", False))
    forecast_df, final_prices, _ = monte_carlo_gbm(
        df["price"],
        pred_end,
        iterations=iterations,
        seed=int(seed) if seed is not None else None,
        use_weekdays_only=use_weekdays,
    )
    summary = summarize_terminal(final_prices)
    spot = float(df["price"].iloc[-1])
    summary["spot"] = spot
    summary["p_above_spot"] = float((final_prices > spot).mean())
    summary["forecast_end"] = pred_end.isoformat()
    summary["ticker"] = ticker
    summary["iterations"] = iterations
    logger.info(
        "Terminal mean=%.2f std=%.2f P(price > spot)=%.1f%%",
        summary["mean"],
        summary["std"],
        100 * summary["p_above_spot"],
    )
    out_cfg = cfg.get("output") or {}
    figures: dict[str, Path] = {}
    if out_cfg.get("save_figures", True):
        figures = run_plots(df, forecast_df, final_prices, ticker, pred_end, iterations, cfg)

    results_path = resolve_project_path(out_cfg.get("results_path", "outputs/results.json"))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data") or {}
    payload = {
        "version": __version__,
        "summary": summary,
        "data_symbol": str(data_cfg.get("ticker", ticker)),
        "figures": {k: path_relative_to_project(v) for k, v in figures.items()},
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", results_path)
    return {
        "forecast_df": forecast_df,
        "final_prices": final_prices,
        "summary": summary,
        "figures": figures,
        "results_path": results_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo GBM price simulation (config-driven)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
