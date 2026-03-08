#!/usr/bin/env python3
"""
Script 07: Train Model with qlib-style Pipeline

Trains LightGBM model using qlib-inspired workflow with Malaysia-specific features.

Usage:
    python scripts/07_qlib_train.py [--ticker 1155] [--universe klci30]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, TRAIN_END_DATE, VALID_START_DATE
from tickers import KLCI30_CODES, get_local_name
from alpha.qlib.handler import BursaDataHandler
from alpha.factors.combiner import compute_all_factors, get_factor_columns
from alpha.factors.opr_regime import load_opr_history


def prepare_dataset(
    tickers: list,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    train_ratio: float = 0.8,
) -> tuple:
    """
    Prepare training dataset.
    
    Args:
        tickers: List of ticker codes
        start_date: Start date
        end_date: End date
        train_ratio: Ratio for train/valid split
    
    Returns:
        (train_df, valid_df, feature_cols)
    """
    print("Loading data...")
    
    # Load OPR history
    opr_history = load_opr_history()
    
    # Load price data
    all_data = []
    
    for ticker in tickers:
        filepath = Path("data/raw/prices") / f"{ticker}.csv"
        
        if not filepath.exists():
            print(f"  ⚠️ {ticker}: No price file")
            continue
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        print(f"  ✅ {ticker}: {len(df)} rows (before factors)")
        
        # Compute all factors
        df = compute_all_factors(df, ticker, opr_history, verbose=False)
        
        # Get factor columns
        factor_cols = get_factor_columns()
        
        # Filter to available columns
        available_cols = [c for c in factor_cols if c in df.columns]
        
        print(f"     {len(available_cols)} features available")
        
        # Add ticker
        df["ticker"] = ticker
        
        all_data.append(df[["date", "ticker", "close"] + available_cols])
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombined: {len(combined)} rows before cleaning")
    
    # Drop palm_oil_beta (all NaN - no FCPO data)
    if "palm_oil_beta" in available_cols:
        available_cols.remove("palm_oil_beta")
        print("  ⚠️  Excluding palm_oil_beta (no FCPO data)")
    
    # Drop only rows with NaN in remaining feature columns
    combined = combined.dropna(subset=available_cols)
    print(f"After cleaning: {len(combined)} rows")
    
    # Date split
    split_date = combined["date"].min() + (combined["date"].max() - combined["date"].min()) * train_ratio
    
    train_df = combined[combined["date"] < split_date].copy()
    valid_df = combined[combined["date"] >= split_date].copy()
    
    # Feature columns
    feature_cols = available_cols
    
    print(f"\nTrain: {len(train_df)} rows ({len(tickers)} tickers)")
    print(f"Valid: {len(valid_df)} rows")
    print(f"Features: {len(feature_cols)}")
    
    return train_df, valid_df, feature_cols


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list,
    params: dict = None,
) -> lgb.Booster:
    """
    Train LightGBM model.
    
    Args:
        train_df: Training data
        valid_df: Validation data
        feature_cols: Feature columns
        params: Model parameters
    
    Returns:
        Trained model
    """
    # Default params
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
    
    # Create target (future 5-day return)
    for df in [train_df, valid_df]:
        df["target"] = df.groupby("ticker")["close"].pct_change(5).shift(-5)
    
    # Remove NaN targets
    train_df = train_df.dropna(subset=["target"])
    valid_df = valid_df.dropna(subset=["target"])
    
    # Create datasets
    train_data = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["target"],
    )
    
    valid_data = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df["target"],
        reference=train_data,
    )
    
    # Train
    print("\nTraining...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    
    return model


def evaluate_model(
    model: lgb.Booster,
    valid_df: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Evaluate model using Information Coefficient (IC).
    
    Args:
        model: Trained model
        valid_df: Validation data
        feature_cols: Feature columns
    
    Returns:
        Evaluation metrics
    """
    # Predict
    valid_df = valid_df.copy()
    valid_df = valid_df.dropna(subset=["target"])
    valid_df["pred"] = model.predict(valid_df[feature_cols])
    
    # IC: correlation between prediction and actual
    ic = valid_df.groupby("date").apply(
        lambda x: x["pred"].corr(x["target"])
    ).mean()
    
    # Rank IC
    rank_ic = valid_df.groupby("date").apply(
        lambda x: x["pred"].rank().corr(x["target"].rank())
    ).mean()
    
    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "n_samples": len(valid_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Train qlib-style model")
    parser.add_argument("--universe", default="klci30", help="Universe to train on")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    args = parser.parse_args()
    
    print("=" * 60)
    print("QLIB-STYLE MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Prepare data
    train_df, valid_df, feature_cols = prepare_dataset(
        tickers=KLCI30_CODES,
        train_ratio=args.train_ratio,
    )
    
    # Train model
    model = train_model(train_df, valid_df, feature_cols)
    
    # Evaluate
    metrics = evaluate_model(model, valid_df, feature_cols)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"IC (Information Coefficient): {metrics['ic']:.4f}")
    print(f"Rank IC: {metrics['rank_ic']:.4f}")
    print(f"Validation samples: {metrics['n_samples']}")
    print(f"Best iteration: {model.best_iteration}")
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)
    
    print(importance.head(10).to_string(index=False))
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = Path("models") / f"qlib_variant_{timestamp}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    print(f"\n✅ Model saved to: {model_path}")


if __name__ == "__main__":
    main()
