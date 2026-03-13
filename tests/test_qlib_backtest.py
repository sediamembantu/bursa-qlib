from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def feature_name(self):
        return ["signal"]

    def predict(self, frame):
        return frame["signal"].to_numpy()


def test_backtest_strategy_does_not_double_count_cash():
    backtest_module = load_script_module("qlib_backtest_script", "scripts/08_qlib_backtest.py")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "ticker": ["1155", "1295", "1155", "1295"],
            "close": [10.0, 20.0, 10.0, 20.0],
            "signal": [0.9, 0.1, 0.9, 0.1],
        }
    )

    nav = backtest_module.backtest_strategy(
        df=df,
        model=FakeModel(),
        initial_capital=1000,
        rebalance_freq=1,
        top_n=1,
    )

    assert not nav.empty
    assert nav["nav"].max() <= 1000
