#!/usr/bin/env python3
"""
Script 01: Fetch all data sources.

This script fetches:
1. Price data from Yahoo Finance (KLCI-30)
2. Macro data from BNM OpenAPI (OPR, FX, KLIBOR)
3. Economic data from OpenDOSM (GDP, CPI, IPI, Trade, Labour)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch import yahoo_finance, bnm_openapi, opendosm


def main():
    """Run all data fetchers."""
    print("=" * 60)
    print("BURSA-QLIB DATA PIPELINE")
    print("Script 01: Fetch All Data")
    print("=" * 60)
    
    # Step 1: Fetch price data
    print("\n[1/3] Fetching Price Data from Yahoo Finance...")
    price_data = yahoo_finance.fetch_universe(universe="klci30")
    if price_data:
        yahoo_finance.save_price_data(price_data)
    else:
        print("WARNING: No price data fetched!")
    
    # Step 2: Fetch BNM macro data
    print("\n[2/3] Fetching Macro Data from BNM OpenAPI...")
    bnm_openapi.main()
    
    # Step 3: Fetch OpenDOSM economic data
    print("\n[3/3] Fetching Economic Data from OpenDOSM...")
    opendosm.main()
    
    print("\n" + "=" * 60)
    print("DATA FETCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
