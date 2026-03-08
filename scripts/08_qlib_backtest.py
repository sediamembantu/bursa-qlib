#!/usr/bin/env python3
"""
Script 08: Backtest qlib Variant

Backtests the qlib-trained model with Bursa Malaysia constraints.

Usage:
    python scripts/08_qlib_backtest.py --model models/qlib_variant_xxx.txt
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

from config import (
    PRICES_DIR,
    LOT_SIZE,
    DAILY_PRICE_LIMIT,
    COMMISSION_RATE,
    STAMP_DUTY_RATE,
    CLEARING_FEE_RATE,
    RANDOM_SEED,
)
from tickers import get_all_tickers, get_local_name
from alpha.factors.combiner import compute_all_factors
from alpha.factors.opr_regime import load_opr_history


def compute_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha158-style features."""
    df = df.copy()
    
    # Returns
    df["return_1d"] = df.groupby("ticker")["close"].pct_change(1)
    df["return_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["return_10d"] = df.groupby("ticker")["close"].pct_change(10)
    df["return_20d"] = df.groupby("ticker")["close"].pct_change(20)
    
    # Moving averages
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


def load_model(model_path: Path):
    """Load trained model."""
    model = lgb.Booster(model_file=str(model_path))
    print(f"Model loaded: {model_path}")
    return model


def load_feature_columns(model_path: Path):
    """Load feature columns for model."""
    feature_file = model_path.parent / model_path.name.replace("qlib_variant", "qlib_features")
    
    if feature_file.exists():
        with open(feature_file) as f:
            return [line.strip() for line in f]
    else:
        # Default features
        return [
            "return_1d", "return_5d", "return_10d", "return_20d",
            "ma_5", "ma_10", "ma_20",
            "vol_5", "vol_10", "vol_20",
            "volume_ratio_5", "volume_ratio_20",
            "high_low_ratio", "close_to_high",
            "momentum_5", "momentum_10", "momentum_20",
            "bb_position", "macd", "macd_signal",
            "shariah_compliant", "glc_flag",
            "cny_window", "hari_raya_window", "deepavali_window", "christmas_window",
            "opr_rate", "opr_hiking", "opr_cutting",
            "fx_sensitivity",
        ]


def calculate_transaction_costs(trade_value: float, is_buy: bool = True) -> dict:
    """Calculate Malaysian transaction costs."""
    # Commission (0.1%, min RM 8)
    commission = max(trade_value * COMMISSION_RATE, 8)
    
    # Stamp duty (0.1%, capped at RM 200)
    stamp_duty = min(trade_value * STAMP_DUTY_RATE, 200)
    
    # Clearing fee (0.03%)
    clearing_fee = trade_value * CLEARING_FEE_RATE
    
    total = commission + stamp_duty + clearing_fee
    
    return {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "clearing_fee": clearing_fee,
        "total": total,
    }


def backtest_strategy(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    initial_capital: float = 1_000_000,
    rebalance_freq: str = "W",
    top_k: int = 10,
):
    """
    Backtest strategy with weekly rebalancing.
    
    Args:
        model: Trained model
        df: DataFrame with features
        feature_cols: Feature columns
        initial_capital: Starting capital
        rebalance_freq: Rebalancing frequency ("W" = weekly)
        top_k: Number of stocks to hold
    
    Returns:
        DataFrame with NAV history
    """
    print("\n" + "=" * 60)
    print("BACKTESTING QLIB VARIANT")
    print("=" * 60)
    
    # Get rebalancing dates
    df_sorted = df.sort_values("date")
    unique_dates = pd.DatetimeIndex(df_sorted["date"].unique())
    
    # Resample to weekly (use last trading day of each week)
    weekly_dates = unique_dates.to_series().resample(rebalance_freq).last().dropna().values
    
    print(f"Rebalancing dates: {len(weekly_dates)}")
    print(f"Initial capital: RM {initial_capital:,.0f}")
    print(f"Top K stocks: {top_k}")
    
    # Initialize
    nav = initial_capital
    cash = initial_capital
    holdings = {}  # ticker -> shares
    nav_history = []
    
    for i, date in enumerate(weekly_dates):
        date = pd.Timestamp(date)
        
        # Get data for this date
        day_data = df[df["date"] == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # Get predictions
        X = day_data[feature_cols].fillna(0)
        predictions = model.predict(X)
        
        day_data["prediction"] = predictions
        
        # Rank by prediction
        ranked = day_data.sort_values("prediction", ascending=False)
        
        # Select top K
        selected = ranked.head(top_k)
        
        # Get current prices
        current_prices = {}
        for _, row in day_data.iterrows():
            current_prices[row["ticker"]] = row["close"]
        
        # Liquidate current holdings
        for ticker, shares in holdings.items():
            if ticker in current_prices:
                price = current_prices[ticker]
                trade_value = shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=False)
                cash += trade_value - costs["total"]
        
        holdings = {}
        
        # Buy new positions
        target_value = cash / top_k
        
        for _, row in selected.iterrows():
            ticker = row["ticker"]
            price = row["close"]
            
            # Apply constraints (Shariah filter)
            if row.get("shariah_compliant", 1) == 0:
                continue
            
            # Calculate shares (100-lot minimum)
            shares = int(target_value / price / LOT_SIZE) * LOT_SIZE
            
            if shares > 0:
                trade_value = shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=True)
                
                if cash >= trade_value + costs["total"]:
                    holdings[ticker] = shares
                    cash -= (trade_value + costs["total"])
        
        # Calculate NAV
        portfolio_value = sum(
            shares * current_prices.get(ticker, 0)
            for ticker, shares in holdings.items()
        )
        nav = cash + portfolio_value
        
        nav_history.append({
            "date": date,
            "nav": nav,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "holdings": len(holdings),
        })
        
        if i % 10 == 0:
            print(f"  {date.strftime('%Y-%m-%d')}: NAV = RM {nav:,.0f}")
    
    # Final liquidation
    final_date = pd.Timestamp(weekly_dates[-1])
    final_data = df[df["date"] == final_date]
    
    for ticker, shares in holdings.items():
        ticker_data = final_data[final_data["ticker"] == ticker]
        if len(ticker_data) > 0:
            price = ticker_data.iloc[0]["close"]
            trade_value = shares * price
            costs = calculate_transaction_costs(trade_value, is_buy=False)
            cash += trade_value - costs["total"]
    
    nav = cash
    
    nav_history.append({
        "date": final_date,
        "nav": nav,
        "cash": cash,
        "portfolio_value": 0,
        "holdings": 0,
    })
    
    return pd.DataFrame(nav_history)


def calculate_metrics(nav_df: pd.DataFrame):
    """Calculate performance metrics."""
    nav_df = nav_df.copy()
    nav_df["return"] = nav_df["nav"].pct_change()
    
    total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1) * 100
    
    # Annualized metrics
    days = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days
    years = days / 365
    
    annual_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) ** (1 / years) - 1
    annual_return *= 100
    
    volatility = nav_df["return"].std() * np.sqrt(52) * 100  # Weekly
    
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    args = parser.parse_args()
    
    print("=" * 60)
    print("QLIB VARIANT BACKTEST")
    print("=" * 60)
    
    # Find latest model
    if args.model:
        model_path = Path(args.model)
    else:
        models_dir = Path(__file__).parent.parent / "models"
        qlib_models = list(models_dir.glob("qlib_variant_*.txt"))
        
        if not qlib_models:
            print("No qlib model found. Run scripts/07_qlib_train.py first.")
            return
        
        model_path = max(qlib_models, key=lambda x: x.stat().st_mtime)
    
    # Load model
    model = load_model(model_path)
    feature_cols = load_feature_columns(model_path)
    
    # Load data
    print("\nLoading data...")
    tickers = get_all_tickers()
    all_data = []
    
    opr_history = load_opr_history()
    
    for ticker in tickers:
        price_file = PRICES_DIR / f"{ticker}.csv"
        if not price_file.exists():
            continue
        
        df = pd.read_csv(price_file)
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        
        df = compute_all_factors(df, ticker, opr_history, verbose=False)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = compute_alpha158_features(combined)
    
    print(f"Loaded: {len(combined)} rows")
    
    # Run backtest
    nav_df = backtest_strategy(
        model,
        combined,
        feature_cols,
        initial_capital=args.capital,
        rebalance_freq="W",
        top_k=10,
    )
    
    # Calculate metrics
    metrics = calculate_metrics(nav_df)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"qlib_nav_{timestamp}.csv"
    nav_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (QLIB VARIANT)")
    print("=" * 60)
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annual Return: {metrics['annual_return']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print("=" * 60)
    print(f"Results saved: {output_file}")


if __name__ == "__main__":
    main()
