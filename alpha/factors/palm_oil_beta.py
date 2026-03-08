"""
Palm Oil Beta Factor

Computes rolling beta of stock returns to FCPO (Crude Palm Oil Futures).
Malaysia is the world's second-largest palm oil producer, so plantation
stocks and related sectors have significant commodity exposure.

Formula:
    beta = Cov(R_stock, R_fcpo) / Var(R_fcpo)

Where returns are computed over a rolling window (default 60 days).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def compute_palm_oil_beta(
    stock_returns: pd.Series,
    fcpo_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling beta to FCPO returns.
    
    Args:
        stock_returns: Daily stock returns (indexed by date)
        fcpo_returns: Daily FCPO returns (indexed by date)
        window: Rolling window in days
    
    Returns:
        Series of rolling beta values
    """
    # Align dates
    aligned = pd.concat([stock_returns, fcpo_returns], axis=1, join='inner')
    aligned.columns = ['stock', 'fcpo']
    
    # Rolling covariance and variance
    cov = aligned['stock'].rolling(window).cov(aligned['fcpo'])
    var = aligned['fcpo'].rolling(window).var()
    
    # Beta = Cov / Var
    beta = cov / var
    
    return beta


def fetch_fcpo_data(
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Fetch FCPO (Crude Palm Oil) futures data.
    
    Uses Yahoo Finance ticker FCPO=F for front-month futures.
    If unavailable, falls back to palm oil ETF or commodity index.
    
    Args:
        start_date: Start date string
        end_date: End date string
        silent: If True, suppress warning messages
    
    Returns:
        DataFrame with date and close columns (empty if unavailable)
    """
    try:
        import yfinance as yf
        
        # Try FCPO futures ticker
        ticker = "FCPO=F"
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, quiet=True)
        
        if df.empty:
            # Fallback: use palm oil related commodity
            ticker = "PALM=F"  # Alternative palm oil ticker
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, quiet=True)
        
        if df.empty:
            # Fallback: Wilmar International (palm oil major)
            ticker = "WLIL.SI"  # Singapore listed
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, quiet=True)
        
        if df.empty:
            if not silent:
                print("  Warning: Palm oil data unavailable, factor set to NaN")
            return pd.DataFrame()
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "close"})
        
        return df[["date", "close"]]
        
    except Exception as e:
        if not silent:
            print(f"  Warning: Could not fetch FCPO data: {e}")
        return pd.DataFrame()


def compute_fcpo_returns(
    fcpo_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily returns from FCPO prices.
    
    Returns:
        Series of daily returns indexed by date
    """
    df = fcpo_df.copy()
    df = df.set_index("date")
    df["return"] = df["close"].pct_change()
    return df["return"].dropna()


def add_palm_oil_beta_factor(
    price_df: pd.DataFrame,
    window: int = 60,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Add palm oil beta factor to price DataFrame.
    
    Args:
        price_df: DataFrame with date, close columns
        window: Rolling window for beta calculation
        silent: If True, suppress messages
    
    Returns:
        DataFrame with added 'palm_oil_beta' column
    """
    # Fetch FCPO data
    fcpo_df = fetch_fcpo_data(
        start_date=str(price_df["date"].min().date()),
        end_date=str(price_df["date"].max().date()),
        silent=silent,
    )
    
    if fcpo_df.empty:
        price_df["palm_oil_beta"] = np.nan
        return price_df
    
    # Compute returns
    fcpo_returns = compute_fcpo_returns(fcpo_df)
    
    stock_returns = price_df.set_index("date")["close"].pct_change()
    
    # Compute rolling beta
    beta = compute_palm_oil_beta(stock_returns, fcpo_returns, window)
    
    # Add to original DataFrame
    price_df = price_df.set_index("date")
    price_df["palm_oil_beta"] = beta
    price_df = price_df.reset_index()
    
    return price_df


if __name__ == "__main__":
    # Test with sample data
    print("Testing palm oil beta factor...")
    
    # Fetch some sample stock data
    import yfinance as yf
    
    # Maybank as example
    stock_df = yf.download("1155.KL", start="2023-01-01", progress=False)
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    stock_df = stock_df.reset_index()
    stock_df = stock_df.rename(columns={"Date": "date", "Close": "close"})
    
    result = add_palm_oil_beta_factor(stock_df)
    
    print(f"Added palm oil beta to {len(result)} rows")
    print(result[["date", "close", "palm_oil_beta"]].tail(10))
