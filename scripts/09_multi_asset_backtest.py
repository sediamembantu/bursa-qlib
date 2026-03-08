#!/usr/bin/env python3
"""
Script 09: Multi-Asset Backtest

Combines equity strategy with bonds and money market for realistic
portfolio comparison vs EPF.

Asset Allocation Options:
- Conservative: 30% equity, 50% bonds, 20% MM
- Balanced: 50% equity, 35% bonds, 15% MM (default)
- Aggressive: 70% equity, 20% bonds, 10% MM

Usage:
    python scripts/09_multi_asset_backtest.py --strategy balanced
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_asset.portfolio import MultiAssetPortfolio, get_epf_benchmark
from config import RANDOM_SEED


STRATEGIES = {
    "conservative": {"equity": 0.30, "bonds": 0.50, "mm": 0.20},
    "balanced": {"equity": 0.50, "bonds": 0.35, "mm": 0.15},
    "aggressive": {"equity": 0.70, "bonds": 0.20, "mm": 0.10},
    "equity_only": {"equity": 1.00, "bonds": 0.00, "mm": 0.00},
}


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


def main():
    parser = argparse.ArgumentParser(description="Multi-asset backtest")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="balanced",
        help="Asset allocation strategy"
    )
    parser.add_argument(
        "--equity-nav",
        type=str,
        default="backtest_results/nav_20260308_1445.csv",
        help="Path to equity NAV file"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-ASSET BACKTEST")
    print("=" * 60)
    print()
    
    # Get strategy weights
    weights = STRATEGIES[args.strategy]
    print(f"Strategy: {args.strategy}")
    print(f"  Equity: {weights['equity']*100:.0f}%")
    print(f"  Bonds: {weights['bonds']*100:.0f}%")
    print(f"  Money Market: {weights['mm']*100:.0f}%")
    print()
    
    # Load equity NAV
    equity_nav_path = Path(args.equity_nav)
    if not equity_nav_path.exists():
        print(f"❌ Equity NAV file not found: {equity_nav_path}")
        print("Run: python scripts/04_backtest.py first")
        return
    
    equity_nav = pd.read_csv(equity_nav_path)
    equity_nav['date'] = pd.to_datetime(equity_nav['date'])
    print(f"Loaded equity NAV: {len(equity_nav)} days")
    print(f"  Period: {equity_nav['date'].min()} to {equity_nav['date'].max()}")
    print(f"  Initial: RM {equity_nav['nav'].iloc[0]:,.2f}")
    print(f"  Final: RM {equity_nav['nav'].iloc[-1]:,.2f}")
    print()
    
    # Create portfolio
    portfolio = MultiAssetPortfolio(
        equity_weight=weights['equity'],
        bond_weight=weights['bonds'],
        money_market_weight=weights['mm'],
    )
    
    # Fetch bond and money market returns
    print("Fetching bond returns...")
    bond_returns = portfolio.fetch_bond_returns(args.start, args.end)
    print(f"  Bond returns: {len(bond_returns)} days")
    
    print("Fetching money market returns...")
    mm_returns = portfolio.fetch_money_market_returns(args.start, args.end)
    print(f"  Money market returns: {len(mm_returns)} days")
    print()
    
    # Combine assets
    print("Combining assets...")
    combined = portfolio.combine_assets(equity_nav, bond_returns, mm_returns)
    print(f"Combined portfolio: {len(combined)} days")
    print()
    
    # Get EPF benchmark
    print("Creating EPF benchmark...")
    epf = get_epf_benchmark(args.start, args.end)
    print(f"EPF benchmark: {len(epf)} days")
    print()
    
    # Calculate metrics
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # Multi-asset portfolio metrics
    ma_metrics = calculate_metrics(combined['portfolio_nav'])
    
    print(f"Multi-Asset Portfolio ({args.strategy}):")
    print(f"  Total Return: {ma_metrics['total_return']:.2f}%")
    print(f"  Annual Return: {ma_metrics['annual_return']:.2f}%")
    print(f"  Volatility: {ma_metrics['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {ma_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {ma_metrics['max_drawdown']:.2f}%")
    print()
    
    # Equity-only metrics
    equity_metrics = calculate_metrics(combined['equity_nav'])
    
    print("Equity-Only (Variant A):")
    print(f"  Total Return: {equity_metrics['total_return']:.2f}%")
    print(f"  Annual Return: {equity_metrics['annual_return']:.2f}%")
    print(f"  Volatility: {equity_metrics['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {equity_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {equity_metrics['max_drawdown']:.2f}%")
    print()
    
    # EPF metrics
    epf_metrics = calculate_metrics(epf['nav'])
    
    print("EPF Benchmark:")
    print(f"  Total Return: {epf_metrics['total_return']:.2f}%")
    print(f"  Annual Return: {epf_metrics['annual_return']:.2f}%")
    print(f"  Volatility: {epf_metrics['volatility']:.4f}%")
    sharpe_str = '∞' if epf_metrics['volatility'] < 0.01 else f"{epf_metrics['sharpe']:.2f}"
    print(f"  Sharpe Ratio: {sharpe_str}")
    print(f"  Max Drawdown: {epf_metrics['max_drawdown']:.4f}%")
    print()
    
    # Comparison table
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print()
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 60)
    print(f"{'EPF':<25} {epf_metrics['total_return']:>9.2f}% {'∞':>10} {epf_metrics['max_drawdown']:>9.4f}%")
    print(f"{'Multi-Asset (' + args.strategy + ')':<25} {ma_metrics['total_return']:>9.2f}% {ma_metrics['sharpe']:>10.2f} {ma_metrics['max_drawdown']:>9.2f}%")
    print(f"{'Equity-Only':<25} {equity_metrics['total_return']:>9.2f}% {equity_metrics['sharpe']:>10.2f} {equity_metrics['max_drawdown']:>9.2f}%")
    print()
    
    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"multi_asset_{args.strategy}_{timestamp}.csv"
    combined.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")
    
    # Analysis
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    if ma_metrics['total_return'] > equity_metrics['total_return']:
        improvement = ma_metrics['total_return'] - equity_metrics['total_return']
        print(f"✅ Multi-asset improves return by {improvement:.2f}%")
    else:
        print(f"⚠️  Multi-asset underperforms equity-only")
    
    if abs(ma_metrics['max_drawdown']) < abs(equity_metrics['max_drawdown']):
        risk_reduction = abs(equity_metrics['max_drawdown']) - abs(ma_metrics['max_drawdown'])
        print(f"✅ Multi-asset reduces max drawdown by {risk_reduction:.2f}%")
    
    if ma_metrics['sharpe'] > equity_metrics['sharpe']:
        print(f"✅ Multi-asset improves Sharpe ratio")
    
    if ma_metrics['total_return'] > epf_metrics['total_return']:
        print(f"🏆 Multi-asset BEATS EPF!")
    else:
        gap = epf_metrics['total_return'] - ma_metrics['total_return']
        print(f"📊 EPF still leads by {gap:.2f}%")


if __name__ == "__main__":
    main()
