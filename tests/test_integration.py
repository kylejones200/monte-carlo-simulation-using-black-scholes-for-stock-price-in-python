from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from monte_carlo_bs.data import load_prices
from monte_carlo_bs.paths import DEFAULT_CONFIG_PATH, DEFAULT_PRICE_CSV, PROJECT_ROOT
from monte_carlo_bs.runner import run


@pytest.fixture
def quick_config(tmp_path: Path) -> Path:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["simulation"]["iterations"] = 25
    cfg["output"]["figure_dpi"] = 72
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_bundled_csv_exists() -> None:
    assert DEFAULT_PRICE_CSV.is_file()


def test_load_prices_from_bundled_csv() -> None:
    df, ticker = load_prices(DEFAULT_CONFIG_PATH)
    assert len(df) > 100
    assert ticker == "USO"
    assert "price" in df.columns


def test_run_pipeline_writes_relative_paths(quick_config: Path) -> None:
    result = run(quick_config)
    results_path = result["results_path"]
    assert results_path.is_file()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["summary"]["iterations"] == 25
    assert payload["data_symbol"] == "USO"

    for rel in payload["figures"].values():
        assert not Path(rel).is_absolute()
        assert (PROJECT_ROOT / rel).is_file()
