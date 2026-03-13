"""Helpers for the repository's qlib-style data workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from alpha.factors.combiner import compute_all_factors, get_factor_columns
from alpha.factors.opr_regime import load_opr_history
from config import PRICES_DIR, QLIB_DIR

REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class BursaDataHandler:
    """Load source CSV data and prepare qlib-style training frames."""

    price_dir: Path = PRICES_DIR
    qlib_dir: Path = QLIB_DIR
    opr_history: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.opr_history is None:
            self.opr_history = load_opr_history()

    def load_price_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a single ticker CSV and apply an optional date filter."""
        filepath = self.price_dir / f"{ticker}.csv"

        if not filepath.exists():
            return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

        df = pd.read_csv(filepath)
        missing_cols = [column for column in REQUIRED_PRICE_COLUMNS if column not in df.columns]
        if missing_cols:
            raise ValueError(f"{filepath} is missing required columns: {missing_cols}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def to_qlib_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Convert source OHLCV data into the repository's qlib-style format."""
        missing_cols = [
            column for column in REQUIRED_PRICE_COLUMNS if column not in price_df.columns
        ]
        if missing_cols:
            raise ValueError(f"Cannot export qlib-style frame; missing columns: {missing_cols}")

        qlib_df = price_df[REQUIRED_PRICE_COLUMNS].copy()
        qlib_df["factor"] = price_df["factor"] if "factor" in price_df.columns else 1.0

        return qlib_df[["date", "open", "high", "low", "close", "volume", "factor"]]

    def export_ticker(self, ticker: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Export one ticker to qlib-style parquet."""
        price_df = self.load_price_data(ticker)
        if price_df.empty:
            return None

        if output_dir is None:
            output_dir = self.qlib_dir / "features"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ticker}.parquet"
        self.to_qlib_frame(price_df).to_parquet(output_path, index=False)
        return output_path

    def write_instruments_file(
        self,
        tickers: Iterable[str],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Write a simple instrument list for downstream workflows."""
        if output_path is None:
            output_path = self.qlib_dir / "instruments" / "klci30.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for ticker in tickers:
            price_df = self.load_price_data(ticker)
            if price_df.empty:
                continue

            lines.append(
                "\t".join(
                    [
                        ticker,
                        price_df["date"].min().strftime("%Y-%m-%d"),
                        price_df["date"].max().strftime("%Y-%m-%d"),
                    ]
                )
            )

        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return output_path

    def available_factor_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return factor columns that are present and not entirely missing."""
        columns = []
        for column in get_factor_columns():
            if column not in frame.columns:
                continue
            if frame[column].isna().all():
                continue
            columns.append(column)
        return columns

    def prepare_training_frame(
        self,
        tickers: Iterable[str],
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        include_factors: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load a multi-ticker training frame and its usable feature columns."""
        frames: list[pd.DataFrame] = []
        feature_columns: set[str] = set()

        for ticker in tickers:
            price_df = self.load_price_data(ticker, start_date=start_date, end_date=end_date)
            if price_df.empty:
                continue

            frame = price_df
            if include_factors:
                frame = compute_all_factors(
                    price_df,
                    ticker,
                    opr_history=self.opr_history,
                    verbose=False,
                )

            frame["ticker"] = ticker
            available_columns = self.available_factor_columns(frame)
            feature_columns.update(available_columns)

            frames.append(frame[["date", "ticker", "close", *available_columns]].copy())

        if not frames:
            return pd.DataFrame(), []

        combined = pd.concat(frames, ignore_index=True, sort=False)
        ordered_feature_columns = [
            column for column in get_factor_columns() if column in feature_columns
        ]

        if ordered_feature_columns:
            combined = combined.dropna(subset=ordered_feature_columns)

        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        return combined, ordered_feature_columns

    @staticmethod
    def add_forward_returns(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
        """Add a forward return target column grouped by ticker."""
        frame = df.copy()
        frame["target"] = (
            frame.groupby("ticker")["close"].shift(-forward_days) / frame["close"] - 1
        )
        return frame
