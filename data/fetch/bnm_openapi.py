"""
BNM OpenAPI data fetcher for Malaysian monetary and FX data.

Endpoints:
- OPR (Overnight Policy Rate) history
- Exchange rates (USD, SGD, CNY)
- KLIBOR (interbank rates)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import (
    MACRO_DIR,
    BNM_API_BASE,
    BNM_ENDPOINTS,
)


BNM_HEADERS = {"Accept": "application/vnd.BNM.API.v1+json"}


def fetch_opr_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OPR (Overnight Policy Rate) history from BNM.
    
    Returns:
        DataFrame with columns: date, rate
    """
    url = f"{BNM_API_BASE}{BNM_ENDPOINTS['opr']}"
    
    print(f"Fetching OPR history from BNM...")
    
    try:
        response = requests.get(url, headers=BNM_HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response
        records = []
        for item in data.get("data", []):
            records.append({
                "date": item.get("effective_date"),
                "rate": float(item.get("opr_rate", 0)),
            })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        print(f"  Fetched {len(df)} OPR records")
        return df
        
    except Exception as e:
        print(f"  Error fetching OPR: {e}")
        return pd.DataFrame()


def fetch_exchange_rate(
    currency: str = "USD",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch exchange rate data from BNM.
    
    Args:
        currency: Currency code (USD, SGD, CNY)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: date, currency, buy, sell, mid
    """
    url = f"{BNM_API_BASE}{BNM_ENDPOINTS['exchange_rate']}/{currency}"
    
    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    
    print(f"Fetching {currency}/MYR exchange rate...")
    
    try:
        response = requests.get(url, headers=BNM_HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response
        records = []
        for item in data.get("data", []):
            rate = item.get("rate", {})
            records.append({
                "date": item.get("date"),
                "currency": currency,
                "buy": float(rate.get("buying_rate", 0)),
                "sell": float(rate.get("selling_rate", 0)),
                "mid": float(rate.get("middle_rate", 0)),
            })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        print(f"  Fetched {len(df)} {currency}/MYR records")
        return df
        
    except Exception as e:
        print(f"  Error fetching {currency}/MYR: {e}")
        return pd.DataFrame()


def fetch_all_exchange_rates(
    currencies: list[str] = ["USD", "SGD", "CNY"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch exchange rates for multiple currencies.
    
    Returns:
        Combined DataFrame with all currency rates
    """
    all_rates = []
    
    for currency in currencies:
        df = fetch_exchange_rate(currency, start_date, end_date)
        if not df.empty:
            all_rates.append(df)
        time.sleep(0.5)  # Rate limiting
    
    if all_rates:
        return pd.concat(all_rates, ignore_index=True)
    return pd.DataFrame()


def fetch_klibor(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch KLIBOR (Kuala Lumpur Interbank Offered Rate) data.
    
    Returns:
        DataFrame with columns: date, tenor, rate
    """
    url = f"{BNM_API_BASE}{BNM_ENDPOINTS['klibor']}"
    
    print(f"Fetching KLIBOR rates...")
    
    try:
        response = requests.get(url, headers=BNM_HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response - KLIBOR has multiple tenors
        records = []
        for item in data.get("data", []):
            date = item.get("date")
            rates = item.get("rate", {})
            
            for tenor, rate in rates.items():
                records.append({
                    "date": date,
                    "tenor": tenor,
                    "rate": float(rate),
                })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "tenor"]).reset_index(drop=True)
        
        print(f"  Fetched {len(df)} KLIBOR records")
        return df
        
    except Exception as e:
        print(f"  Error fetching KLIBOR: {e}")
        return pd.DataFrame()


def save_macro_data(
    df: pd.DataFrame,
    filename: str,
    output_dir: Path = MACRO_DIR,
) -> None:
    """Save macro data to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")


def main():
    """Fetch all BNM macro data."""
    print("=" * 60)
    print("Fetching BNM Macro Data")
    print("=" * 60)
    
    # OPR history
    opr_df = fetch_opr_history()
    if not opr_df.empty:
        save_macro_data(opr_df, "opr_history")
    
    # Exchange rates
    fx_df = fetch_all_exchange_rates()
    if not fx_df.empty:
        save_macro_data(fx_df, "exchange_rates")
    
    # KLIBOR
    klibor_df = fetch_klibor()
    if not klibor_df.empty:
        save_macro_data(klibor_df, "klibor")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
