#!/usr/bin/env python3
"""
Script 07: Train Model with qlib-style Pipeline

Trains LightGBM on the repository's qlib-style feature frame.

Usage:
    python scripts/07_qlib_train.py [--universe klci30]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.qlib.handler import BursaDataHandler
from config import RANDOM_SEED
from tickers import get_all_tickers


def prepare_dataset(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    train_ratio: float = 0.8,
    handler: BursaDataHandler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare train/validation data for the qlib-style workflow."""
    print("Loading data...")

    if handler is None:
        handler = BursaDataHandler()

    combined, feature_cols = handler.prepare_training_frame(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    if combined.empty or not feature_cols:
        raise ValueError(
            "No training data available. Run scripts/01_fetch_data.py first to create price files."
        )

    combined = handler.add_forward_returns(combined, forward_days=5)
    split_date = combined["date"].min() + (
        combined["date"].max() - combined["date"].min()
    ) * train_ratio

    train_df = combined[combined["date"] < split_date].copy()
    valid_df = combined[combined["date"] >= split_date].copy()
    train_df = train_df.dropna(subset=["target"])
    valid_df = valid_df.dropna(subset=["target"])

    if train_df.empty or valid_df.empty:
        raise ValueError("The selected date range does not produce both train and validation data.")

    print(f"\nCombined: {len(combined)} rows after factor cleaning")
    print(f"Train: {len(train_df)} rows ({len(tickers)} tickers)")
    print(f"Valid: {len(valid_df)} rows")
    print(f"Features: {len(feature_cols)}")

    return train_df, valid_df, feature_cols


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> lgb.Booster:
    """Train a LightGBM regressor on the qlib-style frame."""
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": RANDOM_SEED,
        }

    train_data = lgb.Dataset(train_df[feature_cols], label=train_df["target"])
    valid_data = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df["target"],
        reference=train_data,
    )

    print("\nTraining...")
    return lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )


def evaluate_model(
    model: lgb.Booster,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """Evaluate the model using mean IC and rank IC."""
    valid_df = valid_df.copy()
    valid_df["pred"] = model.predict(valid_df[feature_cols])

    ic_by_date = valid_df.groupby("date").apply(lambda x: x["pred"].corr(x["target"]))
    rank_ic_by_date = valid_df.groupby("date").apply(
        lambda x: x["pred"].rank().corr(x["target"].rank())
    )

    return {
        "ic": float(ic_by_date.dropna().mean()),
        "rank_ic": float(rank_ic_by_date.dropna().mean()),
        "n_samples": float(len(valid_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train qlib-style model")
    parser.add_argument("--universe", default="klci30", help="Universe to train on")
    parser.add_argument("--start-date", default="2020-01-01", help="Training window start date")
    parser.add_argument("--end-date", default="2024-12-31", help="Training window end date")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    args = parser.parse_args()

    tickers = get_all_tickers(args.universe)

    print("=" * 60)
    print("QLIB-STYLE MODEL TRAINING")
    print("=" * 60)
    print()

    train_df, valid_df, feature_cols = prepare_dataset(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_ratio=args.train_ratio,
    )
    model = train_model(train_df, valid_df, feature_cols)
    metrics = evaluate_model(model, valid_df, feature_cols)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"IC (Information Coefficient): {metrics['ic']:.4f}")
    print(f"Rank IC: {metrics['rank_ic']:.4f}")
    print(f"Validation samples: {int(metrics['n_samples'])}")
    print(f"Best iteration: {model.best_iteration or model.current_iteration()}")

    print("\nTop 10 Features:")
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = Path("models") / f"qlib_variant_{timestamp}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    print(f"\n[ok] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
