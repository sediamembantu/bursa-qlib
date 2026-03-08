#!/usr/bin/env python3
"""
Script 04: Backtest Strategy

Backtests the trained LightGBM model with Bursa Malaysia specific parameters:
- 100-lot minimum trades
- 30% daily price limits
- Local transaction costs (commission, stamp duty, clearing fee)

Usage:
    python scripts/04_backtest.py [--model models/lightgbm_xxx.txt] [--capital 1000000]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LOT_SIZE,
    DAILY_PRICE_LIMIT,
    COMMISSION_RATE,
    STAMP_DUTY_RATE,
    CLEARING_FEE_RATE,
    PRICES_DIR,
    RANDOM_SEED,
)
from tickers import get_all_tickers, get_local_name
from alpha.factors.combiner import compute_all_factors
from alpha.factors.opr_regime import load_opr_history


# =============================================================================
# Transaction Cost Calculator
# =============================================================================

def calculate_transaction_costs(
    trade_value: float,
    is_buy: bool = True,
) -> dict:
    """
    Calculate Malaysian transaction costs.
    
    Args:
        trade_value: Total trade value in MYR
        is_buy: True for buy, False for sell
    
    Returns:
        Dictionary with cost breakdown
    """
    # Commission (0.1%, min RM 8)
    commission = max(trade_value * COMMISSION_RATE, 8)
    
    # Stamp duty (0.1%, capped at RM 200)
    stamp_duty = min(trade_value * STAMP_DUTY_RATE, 200)
    
    # Clearing fee (0.03%)
    clearing_fee = trade_value * CLEARING_FEE_RATE
    
    # Total
    total = commission + stamp_duty + clearing_fee
    
    return {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "clearing_fee": clearing_fee,
        "total": total,
    }


def adjust_for_lot_size(
    shares: int,
    lot_size: int = LOT_SIZE,
) -> int:
    """
    Round down to nearest lot size.
    
    Args:
        shares: Number of shares
        lot_size: Minimum lot size
    
    Returns:
        Adjusted number of shares
    """
    return (shares // lot_size) * lot_size


# =============================================================================
# Backtest Engine
# =============================================================================

class BursaBacktest:
    """
    Simple backtest engine for Bursa Malaysia.
    
    Features:
    - Daily rebalancing
    - Long-only (no short selling)
    - Transaction costs
    - Position limits
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.25,
        top_n_stocks: int = 10,
    ):
        """
        Initialize backtest.
        
        Args:
            initial_capital: Starting capital in MYR
            max_position_pct: Maximum weight per stock
            max_sector_pct: Maximum weight per sector
            top_n_stocks: Number of stocks to hold
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.top_n = top_n_stocks
        
        self.positions = {}  # ticker -> shares
        self.cash = initial_capital
        self.nav_history = []
        self.trade_history = []
        
    def get_nav(self, prices: dict[str, float]) -> float:
        """Calculate current NAV."""
        holdings_value = sum(
            self.positions.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in self.positions
        )
        return self.cash + holdings_value
    
    def rebalance(
        self,
        date: datetime,
        predictions: dict[str, float],
        prices: dict[str, float],
    ) -> None:
        """
        Rebalance portfolio based on predictions.
        
        Args:
            date: Current date
            predictions: Dictionary of ticker -> predicted return
            prices: Dictionary of ticker -> current price
        """
        # Sort by predicted return (descending)
        sorted_stocks = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top N stocks
        target_stocks = [s[0] for s in sorted_stocks[:self.top_n]]
        
        # Calculate target weights (equal weight)
        target_weight = 1.0 / self.top_n
        
        # Get current NAV
        nav = self.get_nav(prices)
        
        # Sell positions not in target
        for ticker in list(self.positions.keys()):
            if ticker not in target_stocks:
                shares = self.positions[ticker]
                price = prices.get(ticker, 0)
                
                if price > 0 and shares > 0:
                    # Sell all
                    trade_value = shares * price
                    costs = calculate_transaction_costs(trade_value, is_buy=False)
                    
                    self.cash += trade_value - costs["total"]
                    del self.positions[ticker]
                    
                    self.trade_history.append({
                        "date": date,
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": shares,
                        "price": price,
                        "value": trade_value,
                        "costs": costs["total"],
                    })
        
        # Buy/update target positions
        for ticker in target_stocks:
            price = prices.get(ticker, 0)
            
            if price <= 0:
                continue
            
            # Target position value
            target_value = nav * target_weight
            
            # Cap at max position
            target_value = min(target_value, nav * self.max_position_pct)
            
            # Calculate shares
            target_shares = int(target_value / price)
            target_shares = adjust_for_lot_size(target_shares)
            
            current_shares = self.positions.get(ticker, 0)
            
            if target_shares > current_shares:
                # Buy more
                buy_shares = target_shares - current_shares
                trade_value = buy_shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=True)
                
                if self.cash >= trade_value + costs["total"]:
                    self.cash -= trade_value + costs["total"]
                    self.positions[ticker] = target_shares
                    
                    self.trade_history.append({
                        "date": date,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": buy_shares,
                        "price": price,
                        "value": trade_value,
                        "costs": costs["total"],
                    })
            
            elif target_shares < current_shares:
                # Sell some
                sell_shares = current_shares - target_shares
                trade_value = sell_shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=False)
                
                self.cash += trade_value - costs["total"]
                self.positions[ticker] = target_shares
                
                self.trade_history.append({
                    "date": date,
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": sell_shares,
                    "price": price,
                    "value": trade_value,
                    "costs": costs["total"],
                })
        
        # Record NAV
        self.nav_history.append({
            "date": date,
            "nav": self.get_nav(prices),
            "cash": self.cash,
            "holdings": len(self.positions),
        })


def load_model(model_path: str):
    """Load trained LightGBM model."""
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_price_data(
    ticker: str,
    price_dir: Path = PRICES_DIR,
) -> pd.DataFrame:
    """Load price data for a ticker."""
    filepath = price_dir / f"{ticker}.csv"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_backtest(
    model_path: str,
    universe: str = "klci30",
    initial_capital: float = 1_000_000,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> dict:
    """
    Run backtest.
    
    Args:
        model_path: Path to trained model
        universe: Stock universe
        initial_capital: Starting capital
        start_date: Backtest start date
        end_date: Backtest end date
    
    Returns:
        Dictionary with backtest results
    """
    print("=" * 60)
    print("BURSA-QLIB BACKTEST")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path)
    if model is None:
        return {}
    
    # Get tickers
    tickers = get_all_tickers(universe)
    print(f"Universe: {len(tickers)} stocks")
    
    # Load all price data
    print("\nLoading price data...")
    price_data = {}
    for ticker in tickers:
        df = load_price_data(ticker)
        if not df.empty:
            price_data[ticker] = df
    
    print(f"Loaded {len(price_data)} stocks")
    
    # Get common date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Initialize backtest
    backtest = BursaBacktest(
        initial_capital=initial_capital,
        top_n_stocks=10,
    )
    
    # Get trading dates
    first_df = list(price_data.values())[0]
    trading_dates = first_df[
        (first_df["date"] >= start_dt) & (first_df["date"] <= end_dt)
    ]["date"].tolist()
    
    print(f"\nBacktest period: {start_date} to {end_date}")
    print(f"Trading days: {len(trading_dates)}")
    print(f"Initial capital: RM {initial_capital:,.0f}")
    
    # Run backtest
    print("\nRunning backtest...")
    
    opr_history = load_opr_history()
    
    for i, date in enumerate(trading_dates):
        if i % 50 == 0:
            print(f"  Processing {date.strftime('%Y-%m-%d')}...")
        
        # Get predictions for all stocks
        predictions = {}
        prices = {}
        
        for ticker, df in price_data.items():
            # Get price for this date
            row = df[df["date"] == date]
            if row.empty:
                continue
            
            price = row.iloc[0]["close"]
            prices[ticker] = price
            
            # Get features (need historical data up to this date)
            hist_df = df[df["date"] <= date].tail(100)
            if len(hist_df) < 30:
                continue
            
            # Compute factors
            try:
                factor_df = compute_all_factors(hist_df, ticker, opr_history)
                
                # Get latest row with features
                latest = factor_df.iloc[-1]
                
                # Prepare features for prediction - must match training exactly
                # These are the same features used in 03_train_model.py
                feature_cols = [
                    "open", "high", "low", "close", "volume",
                    "daily_return", "volatility_20", "momentum_20", "volume_ratio",
                    "palm_oil_beta", "fx_sensitivity",
                    "shariah_compliant", "shariah_event",
                    "glc_flag", "glc_spread",
                    "cny_window", "hari_raya_window", "deepavali_window",
                    "christmas_window", "year_end", "any_festive",
                    "opr_rate", "opr_regime", "opr_hiking", "opr_cutting", "opr_holding",
                ]
                
                features = {}
                for col in feature_cols:
                    if col in factor_df.columns:
                        val = latest.get(col)
                        if pd.isna(val):
                            # Fill with defaults based on feature type
                            if col in ["open", "high", "low", "close"]:
                                features[col] = price
                            elif col == "volume":
                                features[col] = 0
                            elif col.endswith("_flag") or col.endswith("_compliant"):
                                features[col] = 0
                            elif col.endswith("_window") or col.endswith("_end"):
                                features[col] = 0
                            else:
                                features[col] = 0
                        else:
                            features[col] = float(val)
                    else:
                        # Feature not in dataframe, use default
                        features[col] = 0
                
                # Predict
                X = pd.DataFrame([features])
                pred = model.predict(X)[0]
                predictions[ticker] = pred
                
            except Exception as e:
                continue
        
        # Rebalance
        if predictions:
            backtest.rebalance(date, predictions, prices)
    
    # Calculate results
    nav_df = pd.DataFrame(backtest.nav_history)
    trades_df = pd.DataFrame(backtest.trade_history)
    
    # Performance metrics
    nav_df["return"] = nav_df["nav"].pct_change()
    
    total_return = (nav_df["nav"].iloc[-1] / initial_capital - 1) * 100
    annual_return = total_return * (252 / len(nav_df)) if len(nav_df) > 0 else 0
    volatility = nav_df["return"].std() * np.sqrt(252) * 100 if len(nav_df) > 1 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    max_nav = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - max_nav) / max_nav * 100
    max_drawdown = drawdown.min()
    
    results = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_nav": nav_df["nav"].iloc[-1] if len(nav_df) > 0 else initial_capital,
        "num_trades": len(trades_df),
        "nav_history": nav_df,
        "trade_history": trades_df,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nPerformance:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Annual Return: {annual_return:.2f}%")
    print(f"  Volatility: {volatility:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"\nPortfolio:")
    print(f"  Initial Capital: RM {initial_capital:,.0f}")
    print(f"  Final NAV: RM {results['final_nav']:,.0f}")
    print(f"  Total Trades: {results['num_trades']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest strategy")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--universe", type=str, default="klci30", help="Stock universe")
    args = parser.parse_args()
    
    results = run_backtest(
        model_path=args.model,
        universe=args.universe,
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )
    
    # Save results
    if results:
        output_dir = Path(__file__).parent.parent / "backtest_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if "nav_history" in results and not results["nav_history"].empty:
            nav_path = output_dir / f"nav_{timestamp}.csv"
            results["nav_history"].to_csv(nav_path, index=False)
            print(f"\nNAV history saved to: {nav_path}")
        
        if "trade_history" in results and not results["trade_history"].empty:
            trade_path = output_dir / f"trades_{timestamp}.csv"
            results["trade_history"].to_csv(trade_path, index=False)
            print(f"Trade history saved to: {trade_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
