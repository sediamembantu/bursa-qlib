#!/usr/bin/env python3
"""
Quick comparison of Variant A vs B on 2024-2025 period.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_backtest(nav_path: Path, label: str) -> dict:
    """Analyze backtest results."""
    df = pd.read_csv(nav_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Filter to 2024-2025
    df = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2025-12-31")]
    
    if len(df) == 0:
        return None
    
    df["return"] = df["nav"].pct_change()
    
    total_return = (df["nav"].iloc[-1] / df["nav"].iloc[0] - 1) * 100
    trading_days = len(df)
    annual_return = total_return * (252 / trading_days) if trading_days > 0 else 0
    volatility = df["return"].std() * np.sqrt(252) * 100 if len(df) > 1 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    cummax = df["nav"].cummax()
    drawdown = (df["nav"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    return {
        "variant": label,
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "days": trading_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

print("=" * 60)
print("VARIANT COMPARISON: 2024-2025 PERIOD")
print("=" * 60)
print()

# Find latest backtest files
results_dir = Path("backtest_results")

# Variant B (qlib)
qlib_files = list(results_dir.glob("qlib_nav_*.csv"))
if qlib_files:
    latest_qlib = max(qlib_files, key=lambda x: x.stat().st_mtime)
    qlib_result = analyze_backtest(latest_qlib, "Variant B (qlib)")
    
    if qlib_result:
        print("Variant B (qlib-style):")
        print(f"  Period: {qlib_result['start']} to {qlib_result['end']}")
        print(f"  Trading days: {qlib_result['days']}")
        print(f"  Total Return: {qlib_result['total_return']:.2f}%")
        print(f"  Annual Return: {qlib_result['annual_return']:.2f}%")
        print(f"  Sharpe: {qlib_result['sharpe']:.2f}")
        print(f"  Max DD: {qlib_result['max_drawdown']:.2f}%")
        print()

print("=" * 60)
print("Note: Variant A results on 2024-2025 period not yet available")
print("      Run: python scripts/04_backtest.py --model models/lightgbm_*.txt")
print("            --start 2024-01-01 --end 2025-12-31")
print("=" * 60)
