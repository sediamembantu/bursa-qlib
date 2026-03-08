#!/usr/bin/env python3
"""
Script 03: Train Model

Trains LightGBM model on Bursa data with Alpha158 + Malaysia-specific factors.

Usage:
    python scripts/03_train_model.py [--ticker 1155] [--universe klci30]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RANDOM_SEED,
    TRAIN_END_DATE,
    VALID_START_DATE,
    MODEL_CONFIGS,
    PRICES_DIR,
)
from tickers import get_all_tickers, get_local_name
from alpha.factors.combiner import compute_all_factors, get_factor_columns
from alpha.factors.opr_regime import load_opr_history


def load_price_data(
    ticker: str,
    price_dir: Path = PRICES_DIR,
) -> pd.DataFrame:
    """
    Load price data for a single ticker.
    
    Args:
        ticker: Stock code
        price_dir: Directory containing price CSVs
    
    Returns:
        DataFrame with OHLCV data
    """
    filepath = price_dir / f"{ticker}.csv"
    
    if not filepath.exists():
        print(f"Warning: No price file for {ticker}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def prepare_features_and_labels(
    df: pd.DataFrame,
    forward_days: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and labels for model training.
    
    Args:
        df: DataFrame with price and factor data
        forward_days: Days forward for return calculation
    
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    df = df.copy()
    
    # Compute forward return (label)
    df["forward_return"] = df["close"].shift(-forward_days) / df["close"] - 1
    
    # Get feature columns
    factor_cols = get_factor_columns()
    
    # Add basic price features
    price_features = ["open", "high", "low", "close", "volume"]
    
    # Add derived features
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    
    derived_features = ["daily_return", "volatility_20", "momentum_20", "volume_ratio"]
    
    # Combine all features
    all_features = price_features + derived_features + factor_cols
    available_features = [c for c in all_features if c in df.columns]
    
    # Create feature matrix
    X = df[available_features].copy()
    
    # Create labels
    y = df["forward_return"].copy()
    
    # Remove rows with NaN labels (future data)
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame = None,
    y_valid: pd.Series = None,
    config: dict = None,
):
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features (optional)
        y_valid: Validation labels (optional)
        config: Model configuration
    
    Returns:
        Trained model
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("Error: lightgbm not installed. Run: pip install lightgbm")
        return None
    
    if config is None:
        config = MODEL_CONFIGS["lightgbm"]
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if X_valid is not None and y_valid is not None:
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        valid_sets.append(valid_data)
        valid_names.append("valid")
    
    # Train model
    print("Training LightGBM model...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(X_train.columns)}")
    
    model = lgb.train(
        config,
        train_data,
        num_boost_round=config.get("n_estimators", 200),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50) if len(valid_sets) > 1 else None,
        ],
    )
    
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # IC (Information Coefficient) - correlation between prediction and actual
    ic = np.corrcoef(y_pred, y_test)[0, 1]
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "ic": ic,
        "rmse": np.sqrt(mse),
    }


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--ticker", type=str, help="Single ticker to train on")
    parser.add_argument("--universe", type=str, default="klci30", help="Universe to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BURSA-QLIB MODEL TRAINING")
    print("Script 03: Train Model")
    print("=" * 60)
    
    # Load OPR history
    opr_history = load_opr_history()
    
    # Get tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = get_all_tickers(args.universe)
    
    print(f"\nProcessing {len(tickers)} tickers...")
    
    # Collect all data
    all_data = []
    
    for ticker in tickers:
        print(f"\n{ticker} ({get_local_name(ticker)})")
        
        # Load price data
        price_df = load_price_data(ticker)
        if price_df.empty:
            continue
        
        # Compute factors
        try:
            factor_df = compute_all_factors(price_df, ticker, opr_history)
        except Exception as e:
            print(f"  Error computing factors: {e}")
            continue
        
        # Add ticker column
        factor_df["ticker"] = ticker
        
        all_data.append(factor_df)
    
    if not all_data:
        print("No data available for training!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} rows")
    
    # Prepare features and labels
    X, y = prepare_features_and_labels(combined_df)
    print(f"Features: {X.shape[1]} columns")
    print(f"Samples: {len(X)}")
    
    # Split data
    train_mask = X.index < len(X) * 0.7  # 70% train
    valid_mask = (X.index >= len(X) * 0.7) & (X.index < len(X) * 0.85)  # 15% valid
    test_mask = X.index >= len(X) * 0.85  # 15% test
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Valid: {len(X_valid)}")
    print(f"  Test: {len(X_test)}")
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_valid, y_valid)
    
    if model is None:
        return
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  IC (correlation): {metrics['ic']:.4f}")
    
    # Feature importance
    print(f"\nTop 10 Features:")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # Save model
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
