"""
FX Sensitivity Factor

Computes rolling correlation of stock returns to USD/MYR exchange rate.
Malaysian stocks with significant USD exposure (exporters, GLCs) show
different sensitivities to currency movements.

Formula:
    fx_sensitivity = Corr(R_stock, R_usdmyr) over rolling window

Positive correlation: stock benefits from weaker ringgit (exporters)
Negative correlation: stock benefits from stronger ringgit (importers)
"""

import pandas as pd
import numpy as np
from typing import Optional


def fetch_usdmyr_data(
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch USD/MYR exchange rate data from Yahoo Finance.
    
    Ticker: MYRUSD=X (MYR per USD)
    
    Returns:
        DataFrame with date and rate columns
    """
    try:
        import yfinance as yf
        
        ticker = "MYRUSD=X"
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print("Warning: USD/MYR data not available")
            return pd.DataFrame()
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "rate"})
        
        return df[["date", "rate"]]
        
    except Exception as e:
        print(f"Error fetching USD/MYR data: {e}")
        return pd.DataFrame()


def compute_fx_returns(
    fx_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily FX returns.
    
    Returns:
        Series of daily returns indexed by date
    """
    df = fx_df.copy()
    df = df.set_index("date")
    df["return"] = df["rate"].pct_change()
    return df["return"].dropna()


def compute_fx_sensitivity(
    stock_returns: pd.Series,
    fx_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling correlation to USD/MYR.
    
    Args:
        stock_returns: Daily stock returns
        fx_returns: Daily USD/MYR returns
        window: Rolling window in days
    
    Returns:
        Series of rolling correlations
    """
    # Align dates
    aligned = pd.concat([stock_returns, fx_returns], axis=1, join='inner')
    aligned.columns = ['stock', 'fx']
    
    # Rolling correlation
    correlation = aligned['stock'].rolling(window).corr(aligned['fx'])
    
    return correlation


def add_fx_sensitivity_factor(
    price_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Add FX sensitivity factor to price DataFrame.
    
    Args:
        price_df: DataFrame with date, close columns
        window: Rolling window for correlation
    
    Returns:
        DataFrame with added 'fx_sensitivity' column
    """
    # Fetch USD/MYR data
    fx_df = fetch_usdmyr_data(
        start_date=str(price_df["date"].min().date()),
        end_date=str(price_df["date"].max().date()),
    )
    
    if fx_df.empty:
        price_df["fx_sensitivity"] = np.nan
        return price_df
    
    # Compute returns
    fx_returns = compute_fx_returns(fx_df)
    stock_returns = price_df.set_index("date")["close"].pct_change()
    
    # Compute rolling correlation
    sensitivity = compute_fx_sensitivity(stock_returns, fx_returns, window)
    
    # Add to DataFrame
    price_df = price_df.set_index("date")
    price_df["fx_sensitivity"] = sensitivity
    price_df = price_df.reset_index()
    
    return price_df


if __name__ == "__main__":
    print("Testing FX sensitivity factor...")
    
    import yfinance as yf
    
    # Test with Maybank
    stock_df = yf.download("1155.KL", start="2023-01-01", progress=False)
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    stock_df = stock_df.reset_index()
    stock_df = stock_df.rename(columns={"Date": "date", "Close": "close"})
    
    result = add_fx_sensitivity_factor(stock_df)
    
    print(f"Added FX sensitivity to {len(result)} rows")
    print(result[["date", "close", "fx_sensitivity"]].tail(10))
