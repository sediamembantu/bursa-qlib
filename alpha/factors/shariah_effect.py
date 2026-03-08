"""
Shariah Compliance Effect Factor

Tracks the impact of stocks entering/exiting the Securities Commission
Malaysia's Shariah-compliant securities list.

Shariah-compliant stocks are investable for Islamic funds, which represent
a significant portion of Malaysian institutional capital. Entry/exit from
the list can cause price pressure.

Factor:
    shariah_event: +1 for entry month, -1 for exit month, 0 otherwise
    shariah_compliant: 1 if currently compliant, 0 otherwise
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
from datetime import datetime


# Shariah list update months (May and November each year)
SHARIAH_UPDATE_MONTHS = [5, 11]


def load_shariah_list(
    filepath: Optional[Path] = None,
) -> dict[str, list[str]]:
    """
    Load historical Shariah compliance data.
    
    Returns:
        Dictionary mapping date (YYYY-MM) to list of compliant tickers
    
    Note: In production, this should load from reference/shariah_list.csv
    For now, returns empty dict as placeholder.
    """
    # Placeholder - in production, load from CSV
    # Format: date,ticker,status (in/out)
    if filepath and filepath.exists():
        df = pd.read_csv(filepath)
        result = {}
        for date in df['date'].unique():
            tickers = df[(df['date'] == date) & (df['status'] == 'in')]['ticker'].tolist()
            result[date] = tickers
        return result
    
    return {}


def is_shariah_compliant(
    ticker: str,
    date: datetime,
    shariah_data: dict[str, list[str]],
) -> bool:
    """
    Check if a ticker was Shariah-compliant on a given date.
    
    Args:
        ticker: Stock code
        date: Date to check
        shariah_data: Historical Shariah list data
    
    Returns:
        True if compliant, False otherwise
    """
    # Find the most recent Shariah list before this date
    applicable_month = f"{date.year}-{date.month:02d}"
    
    # Check update months
    if date.month > 5:
        # After May update
        key = f"{date.year}-05"
    else:
        # After November update of previous year
        key = f"{date.year - 1}-11"
    
    if key in shariah_data:
        return ticker in shariah_data[key]
    
    # Default to True for major Malaysian stocks (most are compliant)
    return True


def detect_shariah_event(
    ticker: str,
    current_date: datetime,
    shariah_data: dict[str, list[str]],
    window_months: int = 3,
) -> int:
    """
    Detect if a Shariah status change occurred recently.
    
    Args:
        ticker: Stock code
        current_date: Current date
        shariah_data: Historical Shariah list data
        window_months: Months after event to consider
    
    Returns:
        +1 for recent entry, -1 for recent exit, 0 otherwise
    """
    # This would compare consecutive Shariah lists
    # Placeholder returning 0
    return 0


def add_shariah_factors(
    price_df: pd.DataFrame,
    ticker: str,
    shariah_data: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Add Shariah compliance factors to price DataFrame.
    
    Args:
        price_df: DataFrame with date column
        ticker: Stock code
        shariah_data: Historical Shariah list data
    
    Returns:
        DataFrame with added shariah_compliant and shariah_event columns
    """
    if shariah_data is None:
        shariah_data = load_shariah_list()
    
    df = price_df.copy()
    
    # Default values (most Malaysian stocks are Shariah-compliant)
    df["shariah_compliant"] = 1
    df["shariah_event"] = 0
    
    if shariah_data:
        df["shariah_compliant"] = df["date"].apply(
            lambda d: int(is_shariah_compliant(ticker, pd.to_datetime(d), shariah_data))
        )
        df["shariah_event"] = df["date"].apply(
            lambda d: detect_shariah_event(ticker, pd.to_datetime(d), shariah_data)
        )
    
    return df


# Known Shariah-non-compliant stocks (examples)
NON_COMPLIANT_TICKERS = {
    "4162",  # BAT (British American Tobacco)
    "2629",  # HEINEKEN
    "2836",  # CARLSBG
    "4715",  # GENM (Genting Malaysia - gambling)
    "3182",  # GENTING (gambling)
}


def get_shariah_status(ticker: str) -> bool:
    """
    Quick check for known non-compliant tickers.
    
    Returns:
        True if likely compliant, False if known non-compliant
    """
    return ticker not in NON_COMPLIANT_TICKERS


if __name__ == "__main__":
    print("Testing Shariah factors...")
    
    # Create sample data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "close": np.random.uniform(10, 12, len(dates))})
    
    # Test with known non-compliant ticker
    result = add_shariah_factors(df, "4162")  # BAT
    print(f"BAT shariah_compliant: {result['shariah_compliant'].unique()}")
    
    # Test with compliant ticker
    result = add_shariah_factors(df, "1155")  # Maybank
    print(f"Maybank shariah_compliant: {result['shariah_compliant'].unique()}")
