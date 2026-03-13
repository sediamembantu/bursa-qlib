from __future__ import annotations

import pandas as pd

from alpha.qlib.handler import BursaDataHandler


def write_price_file(price_dir, ticker: str) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "open": [10, 10.2, 10.4, 10.1, 10.5, 10.6, 10.8, 11.0],
            "high": [10.3, 10.4, 10.6, 10.4, 10.7, 10.9, 11.0, 11.2],
            "low": [9.9, 10.0, 10.2, 10.0, 10.3, 10.4, 10.6, 10.8],
            "close": [10.1, 10.3, 10.5, 10.2, 10.6, 10.7, 10.9, 11.1],
            "volume": [1000, 1100, 1050, 980, 1200, 1250, 1300, 1350],
        }
    )
    df.to_csv(price_dir / f"{ticker}.csv", index=False)


def test_export_ticker_writes_qlib_parquet(tmp_path):
    price_dir = tmp_path / "prices"
    output_dir = tmp_path / "qlib" / "features"
    price_dir.mkdir(parents=True)

    write_price_file(price_dir, "1155")
    handler = BursaDataHandler(price_dir=price_dir)

    output_path = handler.export_ticker("1155", output_dir=output_dir)

    assert output_path == output_dir / "1155.parquet"
    exported = pd.read_parquet(output_path)
    assert list(exported.columns) == ["date", "open", "high", "low", "close", "volume", "factor"]
    assert exported["factor"].eq(1.0).all()


def test_prepare_training_frame_uses_available_factor_columns(tmp_path, monkeypatch):
    price_dir = tmp_path / "prices"
    price_dir.mkdir(parents=True)
    write_price_file(price_dir, "1155")

    def fake_compute_all_factors(price_df, ticker, opr_history=None, verbose=False):
        df = price_df.copy()
        df["fx_sensitivity"] = 0.25
        df["shariah_compliant"] = 1
        return df

    monkeypatch.setattr("alpha.qlib.handler.compute_all_factors", fake_compute_all_factors)

    handler = BursaDataHandler(price_dir=price_dir)
    frame, feature_cols = handler.prepare_training_frame(["1155"])

    assert feature_cols == ["fx_sensitivity", "shariah_compliant"]
    assert {"date", "ticker", "close", "fx_sensitivity", "shariah_compliant"} <= set(frame.columns)
    assert frame["ticker"].eq("1155").all()
