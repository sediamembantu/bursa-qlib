"""
GLC Relative Strength Factor

Computes the relative performance of Government-Linked Companies (GLCs)
versus private sector stocks. GLCs in Malaysia often behave differently
due to government policy influence and institutional ownership.

Formula:
    glc_strength = Return(GLC portfolio) - Return(Non-GLC portfolio)

For individual stocks:
    glc_flag: 1 if GLC, 0 otherwise
    glc_spread: GLC portfolio return - market return (if stock is GLC)
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickers import GLC_COMPANIES, is_glc


def get_glc_portfolio_returns(
    price_data: dict[str, pd.DataFrame],
    window: int = 20,
) -> pd.Series:
    """
    Compute equal-weighted GLC portfolio returns.
    
    Args:
        price_data: Dictionary mapping ticker to price DataFrame
        window: Lookback window for return calculation
    
    Returns:
        Series of GLC portfolio daily returns
    """
    glc_returns = []
    
    for ticker, df in price_data.items():
        if ticker in GLC_COMPANIES:
            df = df.copy()
            df["return"] = df["close"].pct_change()
            glc_returns.append(df.set_index("date")["return"])
    
    if not glc_returns:
        return pd.Series()
    
    # Equal-weighted average
    combined = pd.concat(glc_returns, axis=1)
    portfolio_return = combined.mean(axis=1)
    
    return portfolio_return


def get_private_portfolio_returns(
    price_data: dict[str, pd.DataFrame],
    window: int = 20,
) -> pd.Series:
    """
    Compute equal-weighted private sector portfolio returns.
    
    Args:
        price_data: Dictionary mapping ticker to price DataFrame
        window: Lookback window for return calculation
    
    Returns:
        Series of private sector portfolio daily returns
    """
    private_returns = []
    
    for ticker, df in price_data.items():
        if ticker not in GLC_COMPANIES:
            df = df.copy()
            df["return"] = df["close"].pct_change()
            private_returns.append(df.set_index("date")["return"])
    
    if not private_returns:
        return pd.Series()
    
    # Equal-weighted average
    combined = pd.concat(private_returns, axis=1)
    portfolio_return = combined.mean(axis=1)
    
    return portfolio_return


def compute_glc_spread(
    glc_returns: pd.Series,
    private_returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute rolling GLC vs private sector spread.
    
    Args:
        glc_returns: GLC portfolio daily returns
        private_returns: Private sector portfolio daily returns
        window: Rolling window for cumulative return
    
    Returns:
        Series of spread values (GLC - Private)
    """
    # Align dates
    aligned = pd.concat([glc_returns, private_returns], axis=1, join='inner')
    aligned.columns = ['glc', 'private']
    
    # Cumulative returns over window
    glc_cum = (1 + aligned['glc']).rolling(window).apply(np.prod) - 1
    private_cum = (1 + aligned['private']).rolling(window).apply(np.prod) - 1
    
    spread = glc_cum - private_cum
    
    return spread


def add_glc_factors(
    price_df: pd.DataFrame,
    ticker: str,
    glc_spread: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Add GLC-related factors to price DataFrame.
    
    Args:
        price_df: DataFrame with date column
        ticker: Stock code
        glc_spread: Pre-computed GLC spread series (optional)
    
    Returns:
        DataFrame with added glc_flag and glc_spread columns
    """
    df = price_df.copy()
    
    # GLC flag
    df["glc_flag"] = 1 if is_glc(ticker) else 0
    
    # GLC spread (if provided)
    if glc_spread is not None:
        df = df.set_index("date")
        df["glc_spread"] = glc_spread
        df = df.reset_index()
    else:
        df["glc_spread"] = 0.0
    
    return df


if __name__ == "__main__":
    print("Testing GLC factors...")
    
    # Create sample data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    
    # Test GLC ticker (Maybank)
    df = pd.DataFrame({"date": dates, "close": np.random.uniform(10, 12, len(dates))})
    result = add_glc_factors(df, "1155")
    print(f"Maybank glc_flag: {result['glc_flag'].iloc[0]}")
    
    # Test non-GLC ticker
    result = add_glc_factors(df, "0166")  # Inari
    print(f"Inari glc_flag: {result['glc_flag'].iloc[0]}")
    
    # Show GLC companies
    print(f"\nKnown GLCs: {list(GLC_COMPANIES.values())}")
