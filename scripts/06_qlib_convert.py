#!/usr/bin/env python3
"""
Script 06: Convert to qlib Format

Converts existing CSV price data to qlib binary format.

Usage:
    python scripts/06_qlib_convert.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PRICES_DIR
from tickers import KLCI30_CODES


# qlib data paths
QLIB_DATA_DIR = Path(__file__).parent.parent / "data" / "qlib" / "features"


def convert_csv_to_qlib(
    ticker: str,
    price_dir: Path = PRICES_DIR,
    output_dir: Path = QLIB_DATA_DIR,
) -> bool:
    """
    Convert single ticker CSV to qlib binary format.
    
    qlib expects: date, open, high, low, close, volume, factor
    
    Args:
        ticker: Stock code
        price_dir: Input CSV directory
        output_dir: Output binary directory
    
    Returns:
        True if successful
    """
    input_path = price_dir / f"{ticker}.csv"
    
    if not input_path.exists():
        print(f"  ⚠️  {ticker}: No price file")
        return False
    
    # Read CSV
    df = pd.read_csv(input_path)
    
    if len(df) == 0:
        print(f"  ⚠️  {ticker}: Empty file")
        return False
    
    # Ensure date column
    if "date" not in df.columns:
        print(f"  ⚠️  {ticker}: No date column")
        return False
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # qlib expects these columns
    qlib_df = pd.DataFrame()
    qlib_df["date"] = df["date"]
    qlib_df["factor"] = 1.0  # No adjustment factor
    
    # OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            qlib_df[col] = df[col]
        else:
            print(f"  ⚠️  {ticker}: Missing {col}")
            return False
    
    # Save as parquet (qlib can read this)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"
    qlib_df.to_parquet(output_path, index=False)
    
    print(f"  ✅ {ticker}: {len(qlib_df)} days → {output_path.name}")
    return True


def main():
    print("=" * 60)
    print("CONVERT TO QLIB FORMAT")
    print("=" * 60)
    print()
    
    print(f"Input:  {PRICES_DIR}")
    print(f"Output: {QLIB_DATA_DIR}")
    print()
    
    success_count = 0
    
    for ticker in KLCI30_CODES:
        if convert_csv_to_qlib(ticker):
            success_count += 1
    
    print()
    print("=" * 60)
    print(f"Converted {success_count}/{len(KLCI30_CODES)} tickers")
    print("=" * 60)


if __name__ == "__main__":
    main()
