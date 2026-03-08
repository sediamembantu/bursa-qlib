#!/usr/bin/env python3
"""
Script 10: Dynamic Multi-Asset Backtest

Implements Phase 1 improvements:
1. Regime-based allocation (using HMM detector)
2. Monthly rebalancing (reduces transaction costs)
3. Trend following (200-day MA filter)

Expected improvement: +10-15% return
Target: Beat EPF's +13.09%

Usage:
    python scripts/10_dynamic_backtest.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_asset.portfolio import MultiAssetPortfolio, get_epf_benchmark
from config import RANDOM_SEED


def load_regime_labels(filepath: str = "data/processed/regime_labels.csv") -> pd.DataFrame:
    """Load regime labels from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = 'date'
    return df


# Dynamic allocation rules based on regime + trend
ALLOCATION_RULES = {
    # (regime, trend) -> (equity, bonds, mm)
    ("risk_on", "bull"): (0.70, 0.20, 0.10),      # Aggressive
    ("risk_on", "bear"): (0.40, 0.40, 0.20),      # Cautious
    ("risk_off", "bull"): (0.50, 0.35, 0.15),     # Balanced
    ("risk_off", "bear"): (0.30, 0.50, 0.20),     # Conservative
    ("crisis", "bull"): (0.20, 0.55, 0.25),       # Defensive (rare)
    ("crisis", "bear"): (0.10, 0.60, 0.30),       # Crisis mode
    ("recovery", "bull"): (0.60, 0.25, 0.15),     # Opportunistic
    ("recovery", "bear"): (0.30, 0.50, 0.20),     # Cautious recovery
}


def get_trend_signal(prices: pd.Series, window: int = 200) -> str:
    """
    Determine trend based on price vs moving average.
    
    Args:
        prices: Price series
        window: MA window (default 200)
    
    Returns:
        "bull" if above MA, "bear" if below
    """
    ma = prices.rolling(window).mean()
    latest_price = prices.iloc[-1]
    latest_ma = ma.iloc[-1]
    
    return "bull" if latest_price > latest_ma else "bear"


def calculate_metrics(nav_series: pd.Series) -> dict:
    """Calculate performance metrics."""
    returns = nav_series.pct_change()
    
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
    trading_days = len(nav_series)
    annual_return = total_return * (252 / trading_days) if trading_days > 0 else 0
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def dynamic_backtest(
    equity_nav: pd.DataFrame,
    bond_returns: pd.DataFrame,
    mm_returns: pd.DataFrame,
    regime_labels: pd.DataFrame,
    prices_df: pd.DataFrame,
    rebalance_freq: int = 20,  # Monthly
) -> pd.DataFrame:
    """
    Run dynamic multi-asset backtest.
    
    Args:
        equity_nav: Equity NAV history
        bond_returns: Bond returns
        mm_returns: Money market returns
        regime_labels: HMM regime labels
        prices_df: Price data for trend calculation
        rebalance_freq: Rebalancing frequency in days
    
    Returns:
        DataFrame with dynamic portfolio NAV
    """
    # Normalize dates
    equity_nav['date'] = pd.to_datetime(equity_nav['date'])
    bond_returns['date'] = pd.to_datetime(bond_returns['date'])
    mm_returns['date'] = pd.to_datetime(mm_returns['date'])
    regime_labels.index = pd.to_datetime(regime_labels.index)
    
    # Start with equity dates
    combined = equity_nav[['date', 'nav']].copy()
    combined = combined.rename(columns={'nav': 'equity_nav'})
    
    # Normalize equity NAV to start at 1.0
    combined['equity_nav'] = combined['equity_nav'] / combined['equity_nav'].iloc[0]
    
    # Merge bond and money market
    combined = combined.merge(
        bond_returns[['date', 'cumulative_return']],
        on='date',
        how='left'
    )
    combined = combined.rename(columns={'cumulative_return': 'bond_nav_raw'})
    
    combined = combined.merge(
        mm_returns[['date', 'cumulative_return']],
        on='date',
        how='left'
    )
    combined = combined.rename(columns={'cumulative_return': 'mm_nav_raw'})
    
    # Fill forward
    combined['bond_nav_raw'] = combined['bond_nav_raw'].ffill()
    combined['mm_nav_raw'] = combined['mm_nav_raw'].ffill()
    
    # Normalize bond and MM to start at 1.0
    combined['bond_nav'] = combined['bond_nav_raw'] / combined['bond_nav_raw'].iloc[0]
    combined['mm_nav'] = combined['mm_nav_raw'] / combined['mm_nav_raw'].iloc[0]
    
    # Merge regime labels
    regime_labels = regime_labels.reset_index()
    regime_labels.columns = ['date', 'regime', 'regime_name']  # Rename columns
    
    combined = combined.merge(
        regime_labels[['date', 'regime_name']],
        on='date',
        how='left'
    )
    combined['regime_name'] = combined['regime_name'].ffill()
    
    # Calculate KLCI index for trend (equal-weighted average of all stocks)
    if 'klci_index' not in combined.columns:
        # Use equity NAV as proxy for market trend
        combined['klci_index'] = combined['equity_nav']
    
    # Determine trend at each date
    ma_200 = combined['klci_index'].rolling(200).mean()
    combined['trend'] = np.where(combined['klci_index'] > ma_200, "bull", "bear")
    combined['trend'] = combined['trend'].fillna("bull")  # Default to bull
    
    # Get rebalance dates
    dates = combined['date'].sort_values().unique()
    rebalance_dates = set(dates[::rebalance_freq])
    
    # Initialize portfolio
    combined['portfolio_nav'] = 1.0
    combined['equity_weight'] = 0.30  # Start conservative
    combined['bond_weight'] = 0.50
    combined['mm_weight'] = 0.20
    
    current_weights = (0.30, 0.50, 0.20)
    
    # Run backtest
    for i in range(1, len(combined)):
        date = combined.loc[i, 'date']
        
        # Check if rebalance date
        if date in rebalance_dates:
            regime = combined.loc[i, 'regime_name']
            trend = combined.loc[i, 'trend']
            
            # Get allocation from rules
            key = (regime, trend)
            if key in ALLOCATION_RULES:
                current_weights = ALLOCATION_RULES[key]
            
            combined.loc[i, 'equity_weight'] = current_weights[0]
            combined.loc[i, 'bond_weight'] = current_weights[1]
            combined.loc[i, 'mm_weight'] = current_weights[2]
        else:
            # Keep previous weights
            combined.loc[i, 'equity_weight'] = current_weights[0]
            combined.loc[i, 'bond_weight'] = current_weights[1]
            combined.loc[i, 'mm_weight'] = current_weights[2]
        
        # Calculate portfolio return
        equity_ret = combined.loc[i, 'equity_nav'] / combined.loc[i-1, 'equity_nav'] - 1
        bond_ret = combined.loc[i, 'bond_nav'] / combined.loc[i-1, 'bond_nav'] - 1
        mm_ret = combined.loc[i, 'mm_nav'] / combined.loc[i-1, 'mm_nav'] - 1
        
        portfolio_ret = (
            current_weights[0] * equity_ret +
            current_weights[1] * bond_ret +
            current_weights[2] * mm_ret
        )
        
        combined.loc[i, 'portfolio_nav'] = combined.loc[i-1, 'portfolio_nav'] * (1 + portfolio_ret)
    
    return combined


def main():
    print("=" * 60)
    print("DYNAMIC MULTI-ASSET BACKTEST")
    print("=" * 60)
    print()
    print("Phase 1 Improvements:")
    print("  1. Regime-based allocation (HMM detector)")
    print("  2. Monthly rebalancing (reduces costs)")
    print("  3. Trend following (200-day MA)")
    print()
    
    # Load equity NAV
    equity_nav_path = Path("backtest_results/nav_20260308_1445.csv")
    if not equity_nav_path.exists():
        print(f"❌ Equity NAV file not found: {equity_nav_path}")
        return
    
    equity_nav = pd.read_csv(equity_nav_path)
    equity_nav['date'] = pd.to_datetime(equity_nav['date'])
    print(f"Loaded equity NAV: {len(equity_nav)} days")
    
    # Load regime labels
    regime_labels = load_regime_labels()
    print(f"Loaded regime labels: {len(regime_labels)} days")
    
    # Create portfolio
    portfolio = MultiAssetPortfolio()
    
    # Fetch bond and money market returns
    print("Fetching bond returns...")
    bond_returns = portfolio.fetch_bond_returns("2024-01-01", "2025-12-31")
    
    print("Fetching money market returns...")
    mm_returns = portfolio.fetch_money_market_returns("2024-01-01", "2025-12-31")
    
    # Run dynamic backtest
    print("\nRunning dynamic backtest...")
    results = dynamic_backtest(
        equity_nav=equity_nav,
        bond_returns=bond_returns,
        mm_returns=mm_returns,
        regime_labels=regime_labels,
        prices_df=equity_nav,
        rebalance_freq=20,  # Monthly
    )
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # Dynamic portfolio
    dynamic_metrics = calculate_metrics(results['portfolio_nav'])
    
    print("Dynamic Multi-Asset Portfolio:")
    print(f"  Total Return: {dynamic_metrics['total_return']:.2f}%")
    print(f"  Annual Return: {dynamic_metrics['annual_return']:.2f}%")
    print(f"  Volatility: {dynamic_metrics['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {dynamic_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {dynamic_metrics['max_drawdown']:.2f}%")
    print()
    
    # Equity-only for comparison
    equity_metrics = calculate_metrics(results['equity_nav'])
    
    print("Equity-Only (Baseline):")
    print(f"  Total Return: {equity_metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {equity_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {equity_metrics['max_drawdown']:.2f}%")
    print()
    
    # EPF benchmark
    epf = get_epf_benchmark("2024-01-01", "2025-12-31")
    epf_metrics = calculate_metrics(epf['nav'])
    
    print("EPF Benchmark:")
    print(f"  Total Return: {epf_metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: ∞")
    print(f"  Max Drawdown: 0.00%")
    print()
    
    # Comparison table
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 60)
    print(f"{'EPF':<30} {epf_metrics['total_return']:>9.2f}% {'∞':>10} {'0.00%':>10}")
    print(f"{'Dynamic Multi-Asset':<30} {dynamic_metrics['total_return']:>9.2f}% {dynamic_metrics['sharpe']:>10.2f} {dynamic_metrics['max_drawdown']:>9.2f}%")
    print(f"{'Conservative (Static)':<30} {'2.01':>9}% {'0.24':>10} {'-8.57':>9}%")
    print(f"{'Equity-Only':<30} {equity_metrics['total_return']:>9.2f}% {equity_metrics['sharpe']:>10.2f} {equity_metrics['max_drawdown']:>9.2f}%")
    print()
    
    # Analysis
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    if dynamic_metrics['total_return'] > 2.01:
        improvement = dynamic_metrics['total_return'] - 2.01
        print(f"✅ Improves on static conservative by {improvement:.2f}%")
    
    if dynamic_metrics['total_return'] > epf_metrics['total_return']:
        print(f"🏆 BEATS EPF by {dynamic_metrics['total_return'] - epf_metrics['total_return']:.2f}%!")
    else:
        gap = epf_metrics['total_return'] - dynamic_metrics['total_return']
        print(f"📊 Gap to EPF: {gap:.2f}%")
    
    if abs(dynamic_metrics['max_drawdown']) < abs(equity_metrics['max_drawdown']):
        dd_reduction = abs(equity_metrics['max_drawdown']) - abs(dynamic_metrics['max_drawdown'])
        print(f"✅ Reduces max drawdown by {dd_reduction:.2f}%")
    
    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"dynamic_multi_asset_{timestamp}.csv"
    results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
