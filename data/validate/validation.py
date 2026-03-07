"""
Data quality validation for price and macro data.

Checks:
- Zero/negative prices
- High < low inversions
- Extreme daily returns (> 30%)
- Volume anomalies
- Missing data percentage
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PRICES_DIR,
    DAILY_PRICE_LIMIT,
)


def validate_price_data(
    df: pd.DataFrame,
    ticker: str = "unknown",
    daily_limit: float = DAILY_PRICE_LIMIT,
) -> dict:
    """
    Validate price data quality.
    
    Args:
        df: Price DataFrame
        ticker: Ticker symbol for reporting
        daily_limit: Maximum allowed daily return
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "ticker": ticker,
        "total_rows": len(df),
        "issues": [],
        "warnings": [],
    }
    
    if df.empty:
        results["issues"].append("Empty DataFrame")
        return results
    
    # Check for required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        results["issues"].append(f"Missing columns: {missing_cols}")
        return results
    
    # Check for zero/negative prices
    for col in ["open", "high", "low", "close"]:
        zero_count = (df[col] <= 0).sum()
        if zero_count > 0:
            results["issues"].append(f"{col}: {zero_count} zero/negative values")
    
    # Check high < low inversions
    inversions = (df["high"] < df["low"]).sum()
    if inversions > 0:
        results["issues"].append(f"High < Low inversions: {inversions}")
    
    # Check extreme daily returns
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    extreme_returns = (df["daily_return"].abs() > daily_limit).sum()
    if extreme_returns > 0:
        results["warnings"].append(
            f"Extreme returns (>{daily_limit*100:.0f}%): {extreme_returns}"
        )
    
    # Check missing data
    for col in required_cols:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 5:
            results["issues"].append(f"{col}: {missing_pct:.1f}% missing")
        elif missing_pct > 0:
            results["warnings"].append(f"{col}: {missing_pct:.1f}% missing")
    
    # Check volume anomalies (zero volume)
    zero_volume = (df["volume"] == 0).sum()
    if zero_volume > 0:
        results["warnings"].append(f"Zero volume days: {zero_volume}")
    
    return results


def validate_all_price_files(
    price_dir: Path = PRICES_DIR,
) -> list[dict]:
    """
    Validate all price CSV files in a directory.
    
    Returns:
        List of validation results
    """
    results = []
    
    csv_files = list(price_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "combined_prices.csv"]
    
    print(f"Validating {len(csv_files)} price files...")
    
    for filepath in csv_files:
        ticker = filepath.stem
        df = pd.read_csv(filepath)
        result = validate_price_data(df, ticker)
        results.append(result)
        
        # Print issues
        if result["issues"]:
            print(f"  {ticker}: ISSUES - {result['issues']}")
        elif result["warnings"]:
            print(f"  {ticker}: warnings - {result['warnings']}")
        else:
            print(f"  {ticker}: OK")
    
    return results


def generate_validation_report(
    results: list[dict],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate a summary report of validation results.
    
    Args:
        results: List of validation results
        output_path: Optional path to save report
    
    Returns:
        Summary DataFrame
    """
    summary = []
    
    for r in results:
        summary.append({
            "ticker": r["ticker"],
            "total_rows": r["total_rows"],
            "issues_count": len(r["issues"]),
            "warnings_count": len(r["warnings"]),
            "issues": "; ".join(r["issues"]),
            "warnings": "; ".join(r["warnings"]),
            "status": "FAIL" if r["issues"] else "WARN" if r["warnings"] else "OK",
        })
    
    df = pd.DataFrame(summary)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved validation report: {output_path}")
    
    return df


def main():
    """Run data validation."""
    print("=" * 60)
    print("Validating Price Data")
    print("=" * 60)
    
    results = validate_all_price_files()
    report = generate_validation_report(results)
    
    # Summary stats
    total = len(results)
    ok = sum(1 for r in results if not r["issues"] and not r["warnings"])
    warnings = sum(1 for r in results if r["warnings"] and not r["issues"])
    issues = sum(1 for r in results if r["issues"])
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total tickers: {total}")
    print(f"  OK: {ok}")
    print(f"  Warnings: {warnings}")
    print(f"  Issues: {issues}")
    print("=" * 60)


if __name__ == "__main__":
    main()
