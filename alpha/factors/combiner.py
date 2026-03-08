"""
Factor Combiner

Combines all Malaysia-specific factors into a unified feature set
for model training.

Factors included:
1. palm_oil_beta - Rolling beta to FCPO
2. fx_sensitivity - Rolling correlation to USD/MYR
3. shariah_compliant - Shariah compliance flag
4. shariah_event - Entry/exit from Shariah list
5. glc_flag - GLC status
6. glc_spread - GLC vs private sector spread
7. cny_window, hari_raya_window, etc. - Festive seasonality
8. opr_rate, opr_regime - OPR factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alpha.factors.palm_oil_beta import add_palm_oil_beta_factor
from alpha.factors.fx_sensitivity import add_fx_sensitivity_factor
from alpha.factors.shariah_effect import add_shariah_factors
from alpha.factors.glc_strength import add_glc_factors
from alpha.factors.festive_seasonality import add_festive_factors
from alpha.factors.opr_regime import add_opr_factors, load_opr_history


def compute_all_factors(
    price_df: pd.DataFrame,
    ticker: str,
    opr_history: Optional[pd.DataFrame] = None,
    shariah_data: Optional[dict] = None,
    glc_spread: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute all Malaysia-specific factors for a single stock.
    
    Args:
        price_df: DataFrame with date, open, high, low, close, volume columns
        ticker: Stock code
        opr_history: OPR history DataFrame
        shariah_data: Historical Shariah list data
        glc_spread: Pre-computed GLC spread series
    
    Returns:
        DataFrame with all factor columns added
    """
    df = price_df.copy()
    
    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Palm oil beta
    print(f"  Computing palm oil beta for {ticker}...")
    df = add_palm_oil_beta_factor(df, window=60)
    
    # 2. FX sensitivity
    print(f"  Computing FX sensitivity for {ticker}...")
    df = add_fx_sensitivity_factor(df, window=60)
    
    # 3. Shariah factors
    print(f"  Computing Shariah factors for {ticker}...")
    df = add_shariah_factors(df, ticker, shariah_data)
    
    # 4. GLC factors
    print(f"  Computing GLC factors for {ticker}...")
    df = add_glc_factors(df, ticker, glc_spread)
    
    # 5. Festive seasonality
    print(f"  Computing festive seasonality for {ticker}...")
    df = add_festive_factors(df)
    
    # 6. OPR regime
    print(f"  Computing OPR regime for {ticker}...")
    df = add_opr_factors(df, opr_history)
    
    return df


def compute_factors_for_universe(
    price_data: dict[str, pd.DataFrame],
    opr_history: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute factors for all stocks in a universe.
    
    Args:
        price_data: Dictionary mapping ticker to price DataFrame
        opr_history: OPR history DataFrame
    
    Returns:
        Dictionary mapping ticker to DataFrame with factors
    """
    if opr_history is None:
        opr_history = load_opr_history()
    
    results = {}
    
    for ticker, df in price_data.items():
        print(f"\nProcessing {ticker}...")
        try:
            factor_df = compute_all_factors(df, ticker, opr_history)
            results[ticker] = factor_df
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue
    
    return results


def get_factor_columns() -> list[str]:
    """
    Get list of all Malaysia-specific factor column names.
    
    Returns:
        List of factor column names
    """
    return [
        # Commodity & FX
        "palm_oil_beta",
        "fx_sensitivity",
        
        # Regulatory
        "shariah_compliant",
        "shariah_event",
        
        # Ownership
        "glc_flag",
        "glc_spread",
        
        # Seasonality
        "cny_window",
        "hari_raya_window",
        "deepavali_window",
        "christmas_window",
        "year_end",
        "any_festive",
        
        # Monetary policy
        "opr_rate",
        "opr_regime",
        "opr_hiking",
        "opr_cutting",
        "opr_holding",
    ]


def create_factor_summary(
    factor_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Create summary statistics for all factors across universe.
    
    Args:
        factor_data: Dictionary of ticker to factor DataFrame
    
    Returns:
        Summary DataFrame with factor statistics
    """
    factor_cols = get_factor_columns()
    summaries = []
    
    for ticker, df in factor_data.items():
        stats = {"ticker": ticker}
        for col in factor_cols:
            if col in df.columns:
                stats[f"{col}_mean"] = df[col].mean()
                stats[f"{col}_std"] = df[col].std()
                stats[f"{col}_nan_pct"] = df[col].isna().sum() / len(df) * 100
        summaries.append(stats)
    
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    print("Testing factor combiner...")
    
    import yfinance as yf
    
    # Load OPR history
    opr_history = load_opr_history()
    
    # Test with a single stock
    ticker = "1155"  # Maybank
    stock_df = yf.download(f"{ticker}.KL", start="2023-01-01", progress=False)
    
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    
    stock_df = stock_df.reset_index()
    stock_df = stock_df.rename(columns={"Date": "date", "Close": "close"})
    
    print(f"\nComputing factors for {ticker}...")
    factor_df = compute_all_factors(stock_df, ticker, opr_history)
    
    # Show results
    factor_cols = get_factor_columns()
    available_cols = [c for c in factor_cols if c in factor_df.columns]
    
    print(f"\nComputed {len(available_cols)} factors:")
    for col in available_cols:
        print(f"  {col}: mean={factor_df[col].mean():.4f}, nan={factor_df[col].isna().sum()}")
    
    print(f"\nSample rows:")
    print(factor_df[["date", "close"] + available_cols[:5]].tail())
