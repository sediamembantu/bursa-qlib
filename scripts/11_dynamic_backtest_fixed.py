#!/usr/bin/env python3
"""
Script 11: Dynamic Multi-Asset Backtest (Fixed)

Uses actual market data for trend calculation instead of strategy NAV.

Phase 1 Improvements:
1. Regime-based allocation (using HMM detector)
2. Monthly rebalancing (reduces transaction costs)
3. Trend following (200-day MA on KLCI index)

Usage:
    python scripts/11_dynamic_backtest_fixed.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_asset.portfolio import MultiAssetPortfolio, get_epf_benchmark
from config import RANDOM_SEED, PRICES_DIR
from tickers import KLCI30_CODES


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
    ("crisis", "bull"): (0.20, 0.55, 0.25),       # Defensive
    ("crisis", "bear"): (0.10, 0.60, 0.30),       # Crisis mode
    ("recovery", "bull"): (0.60, 0.25, 0.15),     # Opportunistic
    ("recovery", "bear"): (0.30, 0.50, 0.20),     # Cautious recovery
}


def calculate_klci_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calculate KLCI equal-weighted index from price data.
    
    Returns DataFrame with date and index value.
    """
    all_prices = []
    
    for ticker in KLCI30_CODES:
        filepath = PRICES_DIR / f"{ticker}.csv"
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if len(df) > 0:
            all_prices.append(df[['date', 'close']].copy())
    
    if not all_prices:
        return pd.DataFrame()
    
    # Combine all
    combined = pd.concat(all_prices)
    
    # Equal-weighted average by date
    klci = combined.groupby('date')['close'].mean().reset_index()
    klci.columns = ['date', 'klci_index']
    
    # Normalize to 100 at start
    klci['klci_index'] = (klci['klci_index'] / klci['klci_index'].iloc[0]) * 100
    
    return klci


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
    klci_index: pd.DataFrame,
    rebalance_freq: int = 20,  # Monthly
) -> pd.DataFrame:
    """
    Run dynamic multi-asset backtest.
    """
    # Normalize dates
    equity_nav['date'] = pd.to_datetime(equity_nav['date'])
    bond_returns['date'] = pd.to_datetime(bond_returns['date'])
    mm_returns['date'] = pd.to_datetime(mm_returns['date'])
    regime_labels = regime_labels.reset_index()
    regime_labels.columns = ['date', 'regime', 'regime_name']
    regime_labels['date'] = pd.to_datetime(regime_labels['date'])
    klci_index['date'] = pd.to_datetime(klci_index['date'])
    
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
    combined = combined.merge(
        regime_labels[['date', 'regime_name']],
        on='date',
        how='left'
    )
    combined['regime_name'] = combined['regime_name'].ffill()
    
    # Merge KLCI index
    combined = combined.merge(
        klci_index[['date', 'klci_index']],
        on='date',
        how='left'
    )
    
    # Calculate trend from KLCI index (200-day MA)
    ma_200 = combined['klci_index'].rolling(200).mean()
    combined['trend'] = np.where(combined['klci_index'] > ma_200, "bull", "bear")
    combined['trend'] = combined['trend'].fillna("bull")  # Default
    
    # Get rebalance dates
    dates = combined['date'].sort_values().unique()
    rebalance_dates = set(dates[::rebalance_freq])
    
    # Initialize portfolio
    combined['portfolio_nav'] = 1.0
    combined['equity_weight'] = 0.30
    combined['bond_weight'] = 0.50
    combined['mm_weight'] = 0.20
    
    current_weights = (0.30, 0.50, 0.20)
    
    # Run backtest
    for i in range(1, len(combined)):
        date = combined.iloc[i]['date']
        
        # Check if rebalance date
        if date in rebalance_dates:
            regime = combined.iloc[i]['regime_name']
            trend = combined.iloc[i]['trend']
            
            # Get allocation from rules
            key = (regime, trend)
            if key in ALLOCATION_RULES:
                current_weights = ALLOCATION_RULES[key]
            
        # Store current weights
        combined.iloc[i, combined.columns.get_loc('equity_weight')] = current_weights[0]
        combined.iloc[i, combined.columns.get_loc('bond_weight')] = current_weights[1]
        combined.iloc[i, combined.columns.get_loc('mm_weight')] = current_weights[2]
        
        # Calculate portfolio return
        equity_ret = combined.iloc[i]['equity_nav'] / combined.iloc[i-1]['equity_nav'] - 1
        bond_ret = combined.iloc[i]['bond_nav'] / combined.iloc[i-1]['bond_nav'] - 1
        mm_ret = combined.iloc[i]['mm_nav'] / combined.iloc[i-1]['mm_nav'] - 1
        
        portfolio_ret = (
            current_weights[0] * equity_ret +
            current_weights[1] * bond_ret +
            current_weights[2] * mm_ret
        )
        
        combined.iloc[i, combined.columns.get_loc('portfolio_nav')] = \
            combined.iloc[i-1]['portfolio_nav'] * (1 + portfolio_ret)
    
    return combined


def main():
    print("=" * 60)
    print("DYNAMIC MULTI-ASSET BACKTEST (FIXED)")
    print("=" * 60)
    print()
    print("Phase 1 Improvements:")
    print("  1. Regime-based allocation (HMM detector)")
    print("  2. Monthly rebalancing (reduces costs)")
    print("  3. Trend following (200-day MA on KLCI)")
    print()
    
    # Load equity NAV
    equity_nav_path = Path("backtest_results/nav_20260308_1445.csv")
    if not equity_nav_path.exists():
        print(f"❌ Equity NAV file not found: {equity_nav_path}")
        return
    
    equity_nav = pd.read_csv(equity_nav_path)
    print(f"Loaded equity NAV: {len(equity_nav)} days")
    
    # Load regime labels
    regime_labels = load_regime_labels()
    print(f"Loaded regime labels: {len(regime_labels)} days")
    
    # Calculate KLCI index
    print("Calculating KLCI index...")
    klci_index = calculate_klci_index("2024-01-01", "2025-12-31")
    print(f"KLCI index: {len(klci_index)} days")
    
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
        klci_index=klci_index,
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
    
    # Allocation analysis
    print("=" * 60)
    print("ALLOCATION ANALYSIS")
    print("=" * 60)
    print()
    print("Weight distribution:")
    print(results[['equity_weight', 'bond_weight', 'mm_weight']].describe())
    print()
    print("Regime distribution:")
    print(results['regime_name'].value_counts())
    print()
    print("Trend distribution:")
    print(results['trend'].value_counts())
    print()
    
    # Analysis
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    improvement = dynamic_metrics['total_return'] - 2.01
    if improvement > 0:
        print(f"✅ Improves on static conservative by {improvement:.2f}%")
    else:
        print(f"⚠️  Underperforms static conservative by {abs(improvement):.2f}%")
    
    if dynamic_metrics['total_return'] > epf_metrics['total_return']:
        print(f"🏆 BEATS EPF by {dynamic_metrics['total_return'] - epf_metrics['total_return']:.2f}%!")
    else:
        gap = epf_metrics['total_return'] - dynamic_metrics['total_return']
        print(f"📊 Gap to EPF: {gap:.2f}%")
    
    dd_improvement = abs(equity_metrics['max_drawdown']) - abs(dynamic_metrics['max_drawdown'])
    if dd_improvement > 0:
        print(f"✅ Reduces max drawdown by {dd_improvement:.2f}%")
    
    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"dynamic_multi_asset_v2_{timestamp}.csv"
    results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
