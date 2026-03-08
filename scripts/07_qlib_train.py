#!/usr/bin/env python3
"""
Script 07: Train Model with qlib

Trains LightGBM model using qlib framework with Alpha158 + MY factors.

Usage:
    python scripts/07_qlib_train.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, PRICES_DIR
from tickers import get_all_tickers, get_local_name
from alpha.factors.combiner import compute_all_factors, get_factor_columns
from alpha.factors.opr_regime import load_opr_history
from alpha.qlib.expressions import get_expression_dict


def load_all_data():
    """
    Load all ticker data with features.
    
    Since qlib requires complex setup, we'll use a hybrid approach:
    - Use qlib-style expressions (Alpha158 + MY)
    - But train with direct LightGBM (simpler for now)
    """
    print("Loading price data...")
    
    tickers = get_all_tickers()
    all_data = []
    
    # Load OPR history for factors
    opr_history = load_opr_history()
    
    for ticker in tickers:
        price_file = PRICES_DIR / f"{ticker}.csv"
        
        if not price_file.exists():
            print(f"  SKIP {ticker}: No price file")
            continue
        
        # Load price data
        df = pd.read_csv(price_file)
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        
        # Compute Malaysia-specific factors
        df = compute_all_factors(df, ticker, opr_history, verbose=False)
        
        all_data.append(df)
        print(f"  OK {ticker}: {len(df)} rows")
    
    # Combine all
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows, {len(tickers)} tickers")
    
    return combined


def compute_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Alpha158-style features.
    
    Simplified version using pandas.
    """
    df = df.copy()
    
    # Returns
    df["return_1d"] = df.groupby("ticker")["close"].pct_change(1)
    df["return_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["return_10d"] = df.groupby("ticker")["close"].pct_change(10)
    df["return_20d"] = df.groupby("ticker")["close"].pct_change(20)
    
    # Moving averages (normalized)
    df["ma_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean()) / df["close"]
    df["ma_10"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(10).mean()) / df["close"]
    df["ma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean()) / df["close"]
    
    # Volatility
    df["vol_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).std()) / df["close"]
    df["vol_10"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(10).std()) / df["close"]
    df["vol_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).std()) / df["close"]
    
    # Volume ratios
    df["volume_ratio_5"] = df["volume"] / df.groupby("ticker")["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_ratio_20"] = df["volume"] / df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20).mean())
    
    # Price position
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_to_high"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    
    # Momentum
    df["momentum_5"] = df.groupby("ticker")["close"].transform(lambda x: x / x.shift(5))
    df["momentum_10"] = df.groupby("ticker")["close"].transform(lambda x: x / x.shift(10))
    df["momentum_20"] = df.groupby("ticker")["close"].transform(lambda x: x / x.shift(20))
    
    # Bollinger position
    df["bb_mean"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    df["bb_std"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).std())
    df["bb_position"] = (df["close"] - df["bb_mean"]) / (2 * df["bb_std"])
    
    # MACD-like
    df["ema_12"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12).mean())
    df["ema_26"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26).mean())
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())
    
    return df


def create_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create forward return labels."""
    df = df.copy()
    df["label"] = df.groupby("ticker")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )
    return df


def prepare_train_data(df: pd.DataFrame):
    """Prepare data for training."""
    # Feature columns (Alpha158 + MY)
    alpha158_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "ma_5", "ma_10", "ma_20",
        "vol_5", "vol_10", "vol_20",
        "volume_ratio_5", "volume_ratio_20",
        "high_low_ratio", "close_to_high",
        "momentum_5", "momentum_10", "momentum_20",
        "bb_position", "macd", "macd_signal",
    ]
    
    my_cols = [
        "shariah_compliant", "glc_flag",
        "cny_window", "hari_raya_window", "deepavali_window", "christmas_window",
        "opr_rate", "opr_hiking", "opr_cutting",
        "fx_sensitivity",
    ]
    
    feature_cols = alpha158_cols + my_cols
    
    # Filter valid rows
    df_clean = df.dropna(subset=feature_cols + ["label"])
    
    # Time-based split
    train_end = "2024-06-30"
    valid_start = "2024-07-01"
    
    train_df = df_clean[df_clean["date"] <= train_end]
    valid_df = df_clean[df_clean["date"] >= valid_start]
    
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"]
    
    print(f"\nTraining data:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Valid: {len(X_valid)} samples")
    print(f"  Features: {len(feature_cols)}")
    
    return X_train, y_train, X_valid, y_valid, feature_cols


def train_model(X_train, y_train, X_valid, y_valid):
    """Train LightGBM model."""
    print("\nTraining LightGBM model...")
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]
    )
    
    return model


def evaluate_model(model, X_valid, y_valid):
    """Evaluate model using IC."""
    predictions = model.predict(X_valid)
    
    # Information Coefficient (rank correlation)
    ic = np.corrcoef(predictions, y_valid)[0, 1]
    
    print(f"\nValidation IC: {ic:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))
    
    return ic, importance


def save_model(model, feature_cols, output_dir: Path):
    """Save model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_file = output_dir / f"qlib_variant_{timestamp}.txt"
    
    model.save_model(str(model_file))
    
    # Save feature columns
    feature_file = output_dir / f"qlib_features_{timestamp}.txt"
    with open(feature_file, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    
    print(f"\nModel saved: {model_file}")
    print(f"Features saved: {feature_file}")
    
    return model_file


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("QLIB VARIANT TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_all_data()
    
    # Compute Alpha158 features
    print("\nComputing Alpha158 features...")
    df = compute_alpha158_features(df)
    
    # Create labels
    print("Creating labels...")
    df = create_labels(df, horizon=5)
    
    # Prepare train/valid
    X_train, y_train, X_valid, y_valid, feature_cols = prepare_train_data(df)
    
    # Train model
    model = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate
    ic, importance = evaluate_model(model, X_valid, y_valid)
    
    # Save model
    output_dir = Path(__file__).parent.parent / "models"
    model_file = save_model(model, feature_cols, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_file}")
    print(f"IC: {ic:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
