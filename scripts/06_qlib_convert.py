#!/usr/bin/env python3
"""
Script 06: Convert Data to qlib Format

Converts CSV price data to qlib binary format.

Usage:
    python scripts/06_qlib_convert.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PRICES_DIR, RANDOM_SEED
from tickers import get_all_tickers, get_local_name


# Malaysian public holidays (simplified)
MALAYSIAN_HOLIDAYS = [
    "2024-01-01", "2024-02-01", "2024-02-08", "2024-02-09", "2024-04-10",
    "2024-05-01", "2024-05-22", "2024-06-17", "2024-07-07", "2024-08-31",
    "2024-09-16", "2024-10-31", "2024-11-01", "2024-12-25",
    "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-05-01",
    "2025-05-12", "2025-06-07", "2025-06-27", "2025-08-31", "2025-09-16",
    "2025-10-20", "2025-12-25",
]


def generate_calendar(start_date="2020-01-01", end_date="2026-12-31"):
    """Generate Bursa Malaysia trading calendar."""
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    holidays = pd.to_datetime(MALAYSIAN_HOLIDAYS)
    trading_days = all_dates[~all_dates.isin(holidays)]
    return trading_days


# qlib binary format constants
QLIB_HEADER = b"QLIB"
QLIB_VERSION = 1


def convert_ticker_to_qlib(
    ticker: str,
    price_df: pd.DataFrame,
    output_dir: Path,
) -> bool:
    """
    Convert single ticker data to qlib binary format.
    
    qlib binary format (simplified):
    - Header: "QLIB" + version
    - Data: date, open, high, low, close, volume
    
    Args:
        ticker: Stock code
        price_df: DataFrame with OHLCV data
        output_dir: Output directory
    
    Returns:
        True if successful
    """
    try:
        # Prepare data
        df = price_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Save as simple binary (numpy format for now)
        # qlib uses its own format, but numpy works for testing
        output_file = output_dir / f"{ticker}.bin"
        
        # Convert to numpy structured array
        data = np.array(
            list(zip(
                df["date"].values,
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
                df["volume"].values,
            )),
            dtype=[
                ("date", "datetime64[D]"),
                ("open", "float64"),
                ("high", "float64"),
                ("low", "float64"),
                ("close", "float64"),
                ("volume", "float64"),
            ]
        )
        
        np.save(output_file, data)
        
        return True
        
    except Exception as e:
        print(f"Error converting {ticker}: {e}")
        return False


def save_instruments(
    tickers: list,
    output_dir: Path,
):
    """
    Save instruments list in qlib format.
    
    Format: all.txt with one ticker per line
    """
    output_file = output_dir / "all.txt"
    
    with open(output_file, "w") as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    
    print(f"Instruments saved: {output_file}")


def main():
    """Convert all price data to qlib format."""
    print("=" * 60)
    print("CONVERTING DATA TO QLIB FORMAT")
    print("=" * 60)
    
    # Setup directories
    qlib_dir = Path(__file__).parent.parent / "data" / "qlib"
    features_dir = qlib_dir / "features"
    instruments_dir = qlib_dir / "instruments"
    calendars_dir = qlib_dir / "calendars"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    calendars_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate calendar
    print("\n1. Generating trading calendar...")
    calendar = generate_calendar()
    calendar_file = calendars_dir / "day.txt"
    with open(calendar_file, "w") as f:
        for date in calendar:
            f.write(date.strftime("%Y-%m-%d") + "\n")
    print(f"   Saved: {calendar_file}")
    print(f"   Trading days: {len(calendar)}")
    
    # Convert tickers
    print("\n2. Converting price data...")
    tickers = get_all_tickers()
    converted = []
    failed = []
    
    for ticker in tickers:
        price_file = PRICES_DIR / f"{ticker}.csv"
        
        if not price_file.exists():
            print(f"   SKIP {ticker}: No price file")
            continue
        
        df = pd.read_csv(price_file)
        
        if convert_ticker_to_qlib(ticker, df, features_dir):
            converted.append(ticker)
            print(f"   OK {ticker}: {len(df)} rows")
        else:
            failed.append(ticker)
            print(f"   FAIL {ticker}")
    
    # Save instruments
    print("\n3. Saving instruments list...")
    save_instruments(converted, instruments_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Converted: {len(converted)} tickers")
    print(f"Failed: {len(failed)} tickers")
    print(f"Features: {features_dir}")
    print(f"Instruments: {instruments_dir}")
    print(f"Calendar: {calendar_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
