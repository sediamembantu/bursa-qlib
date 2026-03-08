"""
Festive Seasonality Factor

Captures seasonal patterns around major Malaysian holidays:
- Chinese New Year (CNY): January/February
- Hari Raya Aidilfitri: Islamic calendar (shifts yearly)
- Deepavali: October/November
- Christmas: December
- Year-end window dressing: December

Malaysian markets often show distinct patterns around these periods
due to retail investor behavior and institutional window dressing.

Factor:
    festive_window: 1 if within N days of major holiday, 0 otherwise
    year_end: 1 if December, 0 otherwise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# Approximate festive periods (month ranges)
FESTIVE_WINDOWS = {
    "cny": (1, 15, 2, 28),      # Chinese New Year: Jan 15 - Feb 28
    "hari_raya": (3, 1, 5, 15), # Hari Raya: Mar - May (varies by Islamic calendar)
    "deepavali": (10, 15, 11, 15),  # Deepavali: Oct 15 - Nov 15
    "christmas": (12, 15, 12, 31),  # Christmas: Dec 15 - 31
}


def is_in_festive_window(
    date: datetime,
    window_name: str,
) -> bool:
    """
    Check if a date falls within a festive window.
    
    Args:
        date: Date to check
        window_name: Name of festive window
    
    Returns:
        True if within window, False otherwise
    """
    if window_name not in FESTIVE_WINDOWS:
        return False
    
    start_month, start_day, end_month, end_day = FESTIVE_WINDOWS[window_name]
    
    if start_month == end_month:
        # Same month
        return (date.month == start_month and 
                start_day <= date.day <= end_day)
    else:
        # Spans months
        if date.month == start_month:
            return date.day >= start_day
        elif date.month == end_month:
            return date.day <= end_day
        elif start_month < date.month < end_month:
            return True
        return False


def get_hari_raya_date(year: int) -> datetime:
    """
    Estimate Hari Raya date for a given year.
    
    Note: This is an approximation. Actual dates follow Islamic calendar.
    
    Args:
        year: Calendar year
    
    Returns:
        Approximate Hari Raya date
    """
    # Hari Raya dates shift ~11 days earlier each year
    # 2024: April 10, 2025: March 30, 2026: March 20
    base_dates = {
        2024: datetime(2024, 4, 10),
        2025: datetime(2025, 3, 30),
        2026: datetime(2026, 3, 20),
    }
    
    if year in base_dates:
        return base_dates[year]
    
    # Extrapolate
    ref_year = 2024
    ref_date = base_dates[ref_year]
    days_shift = (year - ref_year) * 11
    return ref_date - timedelta(days=days_shift)


def add_festive_factors(
    price_df: pd.DataFrame,
    window_days: int = 15,
) -> pd.DataFrame:
    """
    Add festive seasonality factors to price DataFrame.
    
    Args:
        price_df: DataFrame with date column
        window_days: Days around holiday to include
    
    Returns:
        DataFrame with added festive factor columns
    """
    df = price_df.copy()
    
    # CNY window
    df["cny_window"] = df["date"].apply(
        lambda d: int(is_in_festive_window(pd.to_datetime(d), "cny"))
    )
    
    # Hari Raya window
    df["hari_raya_window"] = df["date"].apply(
        lambda d: int(is_in_festive_window(pd.to_datetime(d), "hari_raya"))
    )
    
    # Deepavali window
    df["deepavali_window"] = df["date"].apply(
        lambda d: int(is_in_festive_window(pd.to_datetime(d), "deepavali"))
    )
    
    # Christmas window
    df["christmas_window"] = df["date"].apply(
        lambda d: int(is_in_festive_window(pd.to_datetime(d), "christmas"))
    )
    
    # Year-end window dressing (December)
    df["year_end"] = df["date"].apply(lambda d: int(d.month == 12))
    
    # Combined festive flag
    df["any_festive"] = (
        df["cny_window"] | 
        df["hari_raya_window"] | 
        df["deepavali_window"] | 
        df["christmas_window"]
    ).astype(int)
    
    return df


if __name__ == "__main__":
    print("Testing festive seasonality factors...")
    
    # Create sample data spanning multiple months
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "close": np.random.uniform(10, 12, len(dates))})
    
    result = add_festive_factors(df)
    
    # Show festive periods
    print("\nCNY period samples:")
    print(result[result["cny_window"] == 1][["date", "cny_window"]].head())
    
    print("\nDecember samples (year-end):")
    print(result[result["year_end"] == 1][["date", "year_end"]].head())
    
    # Summary
    print(f"\nFestive days breakdown:")
    print(f"  CNY: {result['cny_window'].sum()} days")
    print(f"  Hari Raya: {result['hari_raya_window'].sum()} days")
    print(f"  Deepavali: {result['deepavali_window'].sum()} days")
    print(f"  Christmas: {result['christmas_window'].sum()} days")
    print(f"  Year-end: {result['year_end'].sum()} days")
