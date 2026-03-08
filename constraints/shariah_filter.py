"""
Shariah Filter Module

Enforces Shariah-compliant investing rules for Malaysian institutional portfolios.

Shariah Screening Criteria (Securities Commission Malaysia):
1. Core business compliance (no haram activities)
2. Financial ratio screening:
   - Debt/Total Assets < 33%
   - Cash + Interest-bearing securities/Total Assets < 33%
   - Non-permissible income/Total Revenue < 5%

Non-compliant sectors:
- Conventional banking & finance
- Insurance (conventional)
- Gambling/gaming
- Alcohol production/sales
- Pork-related products
- Entertainment (certain categories)
- Tobacco
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tickers import NON_COMPLIANT_TICKERS, get_sector, is_shariah_non_compliant
from config import REFERENCE_DIR


# =============================================================================
# Shariah Screening
# =============================================================================

def load_shariah_list(
    filepath: Optional[Path] = None,
) -> Set[str]:
    """
    Load official Shariah-compliant list from SC Malaysia.
    
    Args:
        filepath: Path to Shariah list CSV
    
    Returns:
        Set of compliant ticker codes
    """
    if filepath is None:
        filepath = REFERENCE_DIR / "shariah_list.csv"
    
    if filepath.exists():
        df = pd.read_csv(filepath)
        return set(df["ticker"].tolist())
    
    # Default: most Malaysian stocks are Shariah-compliant
    # Return empty set to indicate we should use sector-based filtering
    return set()


def is_shariah_compliant(
    ticker: str,
    shariah_list: Optional[Set[str]] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Check if a stock is Shariah-compliant.
    
    Args:
        ticker: Stock code
        shariah_list: Set of compliant tickers (if available)
        sector_mapping: Dictionary mapping ticker to sector
    
    Returns:
        True if compliant, False otherwise
    """
    # Check if in known non-compliant list
    if ticker in NON_COMPLIANT_TICKERS:
        return False
    
    # If we have official list, use it
    if shariah_list is not None:
        return ticker in shariah_list
    
    # Otherwise, use sector-based filtering
    sector = get_sector(ticker) if sector_mapping is None else sector_mapping.get(ticker, "")
    
    # Non-compliant sectors
    non_compliant_sectors = {
        "Gambling",
        "Conventional Banking",
        "Conventional Insurance",
        "Alcohol",
        "Tobacco",
        "Pork",
    }
    
    return sector not in non_compliant_sectors


def apply_shariah_filter(
    tickers: list[str],
    shariah_list: Optional[Set[str]] = None,
) -> list[str]:
    """
    Filter a list of tickers to only Shariah-compliant ones.
    
    Args:
        tickers: List of ticker codes
        shariah_list: Set of compliant tickers
    
    Returns:
        Filtered list of compliant tickers
    """
    return [t for t in tickers if is_shariah_compliant(t, shariah_list)]


# =============================================================================
# Financial Ratio Screening
# =============================================================================

def screen_financial_ratios(
    ticker: str,
    debt_to_assets: float,
    cash_to_assets: float,
    non_permissible_income_ratio: float,
) -> tuple[bool, str]:
    """
    Screen stock based on Shariah financial ratio thresholds.
    
    Args:
        ticker: Stock code
        debt_to_assets: Total debt / Total assets
        cash_to_assets: Cash + interest-bearing / Total assets
        non_permissible_income_ratio: Non-permissible income / Total revenue
    
    Returns:
        Tuple of (is_compliant, reason)
    """
    reasons = []
    
    # Debt ratio check
    if debt_to_assets > 0.33:
        reasons.append(f"Debt ratio {debt_to_assets:.1%} > 33%")
    
    # Cash ratio check
    if cash_to_assets > 0.33:
        reasons.append(f"Cash ratio {cash_to_assets:.1%} > 33%")
    
    # Non-permissible income check
    if non_permissible_income_ratio > 0.05:
        reasons.append(f"Non-permissible income {non_permissible_income_ratio:.1%} > 5%")
    
    is_compliant = len(reasons) == 0
    reason = "; ".join(reasons) if reasons else "Compliant"
    
    return is_compliant, reason


# =============================================================================
# Portfolio-Level Shariah Constraints
# =============================================================================

def enforce_shariah_portfolio(
    weights: Dict[str, float],
    shariah_list: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Enforce Shariah compliance at portfolio level.
    
    Removes non-compliant stocks and redistributes weight.
    
    Args:
        weights: Dictionary of ticker -> weight
        shariah_list: Set of compliant tickers
    
    Returns:
        Adjusted weights dictionary
    """
    # Filter compliant stocks
    compliant = {
        t: w for t, w in weights.items()
        if is_shariah_compliant(t, shariah_list)
    }
    
    # Redistribute weight
    total_weight = sum(compliant.values())
    if total_weight > 0:
        compliant = {t: w / total_weight for t, w in compliant.items()}
    
    return compliant


if __name__ == "__main__":
    print("Shariah Filter Test")
    print("=" * 50)
    
    from tickers import KLCI30_CODES
    
    print("\nKLCI-30 Shariah Compliance Status:")
    for code, name in KLCI30_CODES.items():
        compliant = is_shariah_compliant(code)
        status = "✓ COMPLIANT" if compliant else "✗ NON-COMPLIANT"
        print(f"  {code} ({name}): {status}")
    
    print("\nKnown non-compliant tickers:")
    for ticker in NON_COMPLIANT_TICKERS:
        print(f"  {ticker}")
