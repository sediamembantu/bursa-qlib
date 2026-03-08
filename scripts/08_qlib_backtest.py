#!/usr/bin/env python3
"""
Script 08: Backtest qlib Variant

Backtests the qlib-style model with same constraints as Variant A.

Usage:
    python scripts/08_qlib_backtest.py [--model models/qlib_variant_xxx.txt]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LOT_SIZE,
    COMMISSION_RATE,
    STAMP_DUTY_RATE,
    CLEARING_FEE_RATE,
    RANDOM_SEED,
)
from tickers import KLCI30_CODES
from constraints.shariah_filter import is_shariah_compliant
from constraints.sector_caps import get_sector
from alpha.factors.combiner import compute_all_factors
from alpha.factors.opr_regime import load_opr_history


def load_model(model_path: str) -> lgb.Booster:
    """Load trained model."""
    return lgb.Booster(model_file=model_path)


def calculate_transaction_costs(trade_value: float, is_buy: bool = True) -> dict:
    """Calculate Malaysian transaction costs."""
    commission = max(trade_value * COMMISSION_RATE, 8)
    stamp_duty = min(trade_value * STAMP_DUTY_RATE, 200)
    clearing_fee = trade_value * CLEARING_FEE_RATE
    
    return {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "clearing_fee": clearing_fee,
        "total": commission + stamp_duty + clearing_fee,
    }


def prepare_features(tickers: list, opr_history: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for backtest period.
    
    Returns combined DataFrame with all tickers.
    """
    all_data = []
    
    for ticker in tickers:
        filepath = Path("data/raw/prices") / f"{ticker}.csv"
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Compute factors
        df = compute_all_factors(df, ticker, opr_history, verbose=False)
        
        df["ticker"] = ticker
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove palm_oil_beta
    if "palm_oil_beta" in combined.columns:
        combined = combined.drop(columns=["palm_oil_beta"])
    
    return combined.dropna()


def backtest_strategy(
    df: pd.DataFrame,
    model: lgb.Booster,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,  # Weekly
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Backtest constrained strategy.
    
    Args:
        df: Feature DataFrame
        model: Trained model
        initial_capital: Starting capital
        rebalance_freq: Days between rebalancing
        top_n: Number of stocks to hold
    
    Returns:
        DataFrame with NAV history
    """
    # Get exact feature columns from model
    feature_cols = model.feature_name()
    
    print(f"Using {len(feature_cols)} features")
    print(f"Features: {feature_cols[:5]}...")
    
    # Predict
    df["pred"] = model.predict(df[feature_cols])
    
    # Filter dates (2024 only for comparison)
    df = df[df["date"] >= "2024-01-01"].copy()
    
    # Get rebalance dates
    dates = df["date"].sort_values().unique()
    rebalance_dates = dates[::rebalance_freq]
    
    # Track NAV
    nav_history = []
    capital = initial_capital
    holdings = {}  # ticker -> shares
    
    for i, date in enumerate(dates):
        date_df = df[df["date"] == date].copy()
        
        if date in rebalance_dates:
            # Apply Shariah filter
            date_df["shariah"] = date_df["ticker"].apply(is_shariah_compliant)
            date_df = date_df[date_df["shariah"]].copy()
            
            if len(date_df) == 0:
                continue
            
            # Select top N by prediction
            date_df = date_df.nlargest(top_n, "pred")
            
            # Equal weight
            target_capital = capital / len(date_df)
            
            new_holdings = {}
            
            for _, row in date_df.iterrows():
                ticker = row["ticker"]
                price = row["close"]
                
                # Calculate shares (with lot size)
                shares = int(target_capital / price / LOT_SIZE) * LOT_SIZE
                
                if shares > 0:
                    new_holdings[ticker] = shares
            
            # Calculate trade costs
            for ticker, new_shares in new_holdings.items():
                old_shares = holdings.get(ticker, 0)
                trade_shares = abs(new_shares - old_shares)
                
                if trade_shares > 0:
                    # Find price for this ticker on this date
                    price = date_df[date_df["ticker"] == ticker]["close"].values[0]
                    trade_value = trade_shares * price
                    costs = calculate_transaction_costs(trade_value)
                    capital -= costs["total"]
            
            holdings = new_holdings
        
        # Calculate current portfolio value
        portfolio_value = capital
        
        for ticker, shares in holdings.items():
            ticker_df = date_df[date_df["ticker"] == ticker]
            if len(ticker_df) > 0:
                price = ticker_df["close"].values[0]
                portfolio_value += shares * price
        
        nav_history.append({
            "date": date,
            "nav": portfolio_value,
            "n_holdings": len(holdings),
        })
    
    return pd.DataFrame(nav_history)


def main():
    parser = argparse.ArgumentParser(description="Backtest qlib variant")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    args = parser.parse_args()
    
    print("=" * 60)
    print("QLIB VARIANT BACKTEST")
    print("=" * 60)
    print()
    
    # Find latest model
    if args.model:
        model_path = Path(args.model)
    else:
        models = list(Path("models").glob("qlib_variant_*.txt"))
        if not models:
            print("❌ No qlib variant model found")
            print("Run: python scripts/07_qlib_train.py")
            return
        model_path = max(models, key=lambda x: x.stat().st_mtime)
    
    print(f"Model: {model_path}")
    
    # Load model
    model = load_model(str(model_path))
    
    # Load data
    print("\nLoading data...")
    opr_history = load_opr_history()
    df = prepare_features(KLCI30_CODES, opr_history)
    
    print(f"Features: {len(df)} rows")
    
    # Backtest
    print("\nRunning backtest...")
    nav_df = backtest_strategy(df, model, args.capital)
    
    # Calculate metrics
    nav_df["return"] = nav_df["nav"].pct_change()
    
    total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1) * 100
    
    trading_days = len(nav_df)
    annual_return = total_return * (252 / trading_days) if trading_days > 0 else 0
    
    volatility = nav_df["return"].std() * np.sqrt(252) * 100 if len(nav_df) > 1 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Volatility: {volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Trading Days: {trading_days}")
    
    # Save results
    output_path = Path("backtest_results") / f"qlib_nav_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON vs VARIANT A")
    print("=" * 60)
    print("| Metric      | Variant A | Variant B |")
    print("|-------------|-----------|-----------|")
    print(f"| Return      | +4.74%    | {total_return:+.2f}%    |")
    print(f"| Sharpe      | 0.37      | {sharpe:.2f}      |")
    print(f"| Max DD      | -4.86%    | {max_drawdown:.2f}%    |")


if __name__ == "__main__":
    main()
