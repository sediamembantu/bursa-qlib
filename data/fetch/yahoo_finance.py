"""
Yahoo Finance data fetcher for Bursa Malaysia OHLCV data.
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    PRICES_DIR,
    START_DATE,
    YF_BATCH_SIZE,
    YF_SLEEP_SECONDS,
)
from tickers import get_yahoo_ticker, get_all_tickers


def fetch_single_ticker(
    ticker: str,
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single Bursa ticker.
    
    Args:
        ticker: Bursa ticker (without .KL suffix)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    
    Returns:
        DataFrame with OHLCV data
    """
    yf_ticker = get_yahoo_ticker(ticker)
    
    print(f"Fetching {yf_ticker}...")
    
    try:
        df = yf.download(
            yf_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        
        if df.empty:
            print(f"  Warning: No data returned for {yf_ticker}")
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Standardise column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        
        # Reset index to get date as column
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        
        # Add ticker column
        df["ticker"] = ticker
        
        # Calculate adjustment factor
        if "adj_close" in df.columns and "close" in df.columns:
            df["factor"] = df["adj_close"] / df["close"]
        
        print(f"  Fetched {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"  Error fetching {yf_ticker}: {e}")
        return pd.DataFrame()


def fetch_universe(
    universe: str = "klci30",
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    batch_size: int = YF_BATCH_SIZE,
    sleep_seconds: float = YF_SLEEP_SECONDS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers in a universe.
    
    Args:
        universe: Universe name ("klci30", "extended", "all")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        batch_size: Number of tickers before sleep
        sleep_seconds: Sleep time between batches
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    tickers = get_all_tickers(universe)
    results = {}
    
    print(f"Fetching {len(tickers)} tickers for universe: {universe}")
    
    for i, ticker in enumerate(tickers):
        df = fetch_single_ticker(ticker, start_date, end_date)
        if not df.empty:
            results[ticker] = df
        
        # Rate limiting
        if (i + 1) % batch_size == 0 and i < len(tickers) - 1:
            print(f"  Sleeping {sleep_seconds}s...")
            time.sleep(sleep_seconds)
    
    print(f"Fetched {len(results)} / {len(tickers)} tickers successfully")
    return results


def save_price_data(
    data: dict[str, pd.DataFrame],
    output_dir: Path = PRICES_DIR,
    combined: bool = True,
) -> None:
    """
    Save fetched price data to CSV files.
    
    Args:
        data: Dictionary mapping ticker to DataFrame
        output_dir: Output directory
        combined: If True, also save combined CSV with all tickers
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual files
    for ticker, df in data.items():
        filepath = output_dir / f"{ticker}.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath}")
    
    # Save combined file
    if combined:
        combined_df = pd.concat(data.values(), ignore_index=True)
        combined_path = output_dir / "combined_prices.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined file: {combined_path}")


def main():
    """Fetch KLCI-30 price data."""
    print("=" * 60)
    print("Fetching Bursa Malaysia Price Data")
    print("=" * 60)
    
    # Fetch KLCI-30
    data = fetch_universe(universe="klci30")
    
    # Save to CSV
    save_price_data(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
