"""
BNM OpenAPI data fetcher for Malaysian monetary and FX data.

Endpoints:
- OPR (Overnight Policy Rate) history
- Exchange rates (USD, SGD, CNY, EUR, etc.)
- KLIBOR (interbank rates) - endpoint may have changed
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
    
    Note: BNM API returns current OPR. For historical data,
    we need to use the year endpoint or store incrementally.
    
    Returns:
        DataFrame with columns: date, rate
    """
    url = f"{BNM_API_BASE}{BNM_ENDPOINTS['opr']}"
    
    print(f"Fetching OPR history from BNM...")
    
    try:
        response = requests.get(url, headers=BNM_HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # API returns current OPR, need to fetch historical by year
        # For now, store current rate
        records = []
        current = data.get("data", {})
        records.append({
            "date": current.get("date"),
            "rate": float(current.get("new_opr_level", 0)),
            "change": float(current.get("change_in_opr", 0)),
            "year": current.get("year"),
        })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        print(f"  Fetched {len(df)} OPR records (current)")
        return df
        
    except Exception as e:
        print(f"  Error fetching OPR: {e}")
        return pd.DataFrame()


def fetch_all_exchange_rates() -> pd.DataFrame:
    """
    Fetch all exchange rates from BNM.
    
    Returns:
        DataFrame with columns: date, currency, buy, sell, mid
    """
    url = f"{BNM_API_BASE}{BNM_ENDPOINTS['exchange_rate']}"
    
    print(f"Fetching exchange rates from BNM...")
    
    try:
        response = requests.get(url, headers=BNM_HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response
        records = []
        for item in data.get("data", []):
            rate = item.get("rate", {})
            
            # Handle null/missing rates
            buy = rate.get("buying_rate")
            sell = rate.get("selling_rate")
            mid = rate.get("middle_rate")
            
            if buy is None or sell is None or mid is None:
                continue
            
            records.append({
                "date": rate.get("date"),
                "currency": item.get("currency_code"),
                "unit": item.get("unit", 1),
                "buy": float(buy),
                "sell": float(sell),
                "mid": float(mid),
            })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "currency"]).reset_index(drop=True)
        
        # Filter for main currencies
        main_currencies = ["USD", "SGD", "CNY", "EUR", "GBP", "JPY", "AUD"]
        df = df[df["currency"].isin(main_currencies)]
        
        print(f"  Fetched {len(df)} exchange rate records")
        return df
        
    except Exception as e:
        print(f"  Error fetching exchange rates: {e}")
        return pd.DataFrame()


def fetch_klibor() -> pd.DataFrame:
    """
    Fetch KLIBOR (Kuala Lumpur Interbank Offered Rate) data.
    
    Note: Endpoint may have changed. Returns empty if unavailable.
    
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
            
            if isinstance(rates, dict):
                for tenor, rate in rates.items():
                    if isinstance(rate, (int, float)):
                        records.append({
                            "date": date,
                            "tenor": tenor,
                            "rate": float(rate),
                        })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["date", "tenor"]).reset_index(drop=True)
        
        print(f"  Fetched {len(df)} KLIBOR records")
        return df
        
    except Exception as e:
        print(f"  Warning: KLIBOR endpoint unavailable ({e})")
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
    
    # KLIBOR (may be unavailable)
    klibor_df = fetch_klibor()
    if not klibor_df.empty:
        save_macro_data(klibor_df, "klibor")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
