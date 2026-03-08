"""
Sector Constraints Module

Enforces sector concentration limits for institutional portfolios.

Malaysian Pension Fund Guidelines (EPF/KWAP style):
- Maximum single sector weight: 25%
- Maximum financial sector: 30% (higher due to market composition)
- Maximum single stock: 10%
- Minimum number of stocks: 15 (diversification)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tickers import SECTOR_MAPPING, get_sector
from config import MAX_SECTOR_WEIGHT, MAX_STOCK_WEIGHT


# =============================================================================
# Sector Concentration Limits
# =============================================================================

# Sector-specific limits (more conservative for institutional)
SECTOR_LIMITS = {
    "Financials": 0.30,        # Higher due to Bursa composition
    "Telecommunications": 0.20,
    "Plantations": 0.20,
    "Consumer Staples": 0.15,
    "Consumer Discretionary": 0.15,
    "Energy": 0.15,
    "Utilities": 0.15,
    "Technology": 0.15,
    "Healthcare": 0.15,
    "Industrials": 0.20,
    "Real Estate": 0.15,
}

# Default limit for unspecified sectors
DEFAULT_SECTOR_LIMIT = 0.15


def get_sector_limit(sector: str) -> float:
    """Get maximum weight for a sector."""
    return SECTOR_LIMITS.get(sector, DEFAULT_SECTOR_LIMIT)


def calculate_sector_weights(
    stock_weights: Dict[str, float],
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Calculate aggregate sector weights from stock weights.
    
    Args:
        stock_weights: Dictionary of ticker -> weight
        sector_mapping: Dictionary mapping ticker to sector
    
    Returns:
        Dictionary of sector -> total weight
    """
    if sector_mapping is None:
        sector_mapping = SECTOR_MAPPING
    
    sector_weights = {}
    
    for ticker, weight in stock_weights.items():
        sector = get_sector(ticker) if sector_mapping is None else sector_mapping.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return sector_weights


def check_sector_constraints(
    stock_weights: Dict[str, float],
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if portfolio meets sector concentration limits.
    
    Args:
        stock_weights: Dictionary of ticker -> weight
        sector_mapping: Dictionary mapping ticker to sector
    
    Returns:
        Tuple of (is_valid, sector_weights)
    """
    sector_weights = calculate_sector_weights(stock_weights, sector_mapping)
    
    # Check each sector
    is_valid = True
    for sector, weight in sector_weights.items():
        limit = get_sector_limit(sector)
        if weight > limit:
            is_valid = False
            break
    
    return is_valid, sector_weights


def enforce_sector_limits(
    stock_weights: Dict[str, float],
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Enforce sector concentration limits by reducing overweight positions.
    
    Args:
        stock_weights: Dictionary of ticker -> weight
        sector_mapping: Dictionary mapping ticker to sector
    
    Returns:
        Adjusted stock weights
    """
    if sector_mapping is None:
        sector_mapping = SECTOR_MAPPING
    
    weights = stock_weights.copy()
    
    # Iterate until all constraints satisfied
    max_iterations = 10
    for _ in range(max_iterations):
        sector_weights = calculate_sector_weights(weights, sector_mapping)
        
        # Find overweight sectors
        overweight = {}
        for sector, weight in sector_weights.items():
            limit = get_sector_limit(sector)
            if weight > limit:
                overweight[sector] = weight - limit
        
        if not overweight:
            break
        
        # Reduce positions in overweight sectors
        for sector, excess in overweight.items():
            # Find stocks in this sector
            sector_stocks = [
                t for t, w in weights.items()
                if get_sector(t) == sector
            ]
            
            if not sector_stocks:
                continue
            
            # Reduce proportionally
            total_sector_weight = sum(weights.get(t, 0) for t in sector_stocks)
            if total_sector_weight > 0:
                reduction_factor = (total_sector_weight - excess) / total_sector_weight
                
                for ticker in sector_stocks:
                    if ticker in weights:
                        weights[ticker] *= reduction_factor
    
    # Renormalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    
    return weights


# =============================================================================
# Single Stock Limits
# =============================================================================

def enforce_stock_limits(
    stock_weights: Dict[str, float],
    max_weight: float = MAX_STOCK_WEIGHT,
) -> Dict[str, float]:
    """
    Enforce single stock weight limits.
    
    Args:
        stock_weights: Dictionary of ticker -> weight
        max_weight: Maximum weight per stock
    
    Returns:
        Adjusted weights
    """
    weights = stock_weights.copy()
    
    # Cap each position
    capped = {}
    excess = 0.0
    
    for ticker, weight in weights.items():
        if weight > max_weight:
            capped[ticker] = max_weight
            excess += weight - max_weight
        else:
            capped[ticker] = weight
    
    # Redistribute excess proportionally to uncapped stocks
    if excess > 0:
        uncapped_total = sum(w for t, w in capped.items() if w < max_weight)
        
        if uncapped_total > 0:
            for ticker in capped:
                if capped[ticker] < max_weight:
                    capped[ticker] += excess * (capped[ticker] / uncapped_total)
    
    return capped


# =============================================================================
# Diversification Requirements
# =============================================================================

def check_diversification(
    stock_weights: Dict[str, float],
    min_stocks: int = 15,
    min_weight: float = 0.01,
) -> Tuple[bool, int]:
    """
    Check if portfolio meets diversification requirements.
    
    Args:
        stock_weights: Dictionary of ticker -> weight
        min_stocks: Minimum number of stocks
        min_weight: Minimum weight to count as a position
    
    Returns:
        Tuple of (is_diversified, number_of_positions)
    """
    active_positions = sum(1 for w in stock_weights.values() if w >= min_weight)
    
    return active_positions >= min_stocks, active_positions


# =============================================================================
# Full Constraint Enforcement
# =============================================================================

def enforce_all_constraints(
    stock_weights: Dict[str, float],
    shariah_compliant_only: bool = True,
    shariah_list: Optional[set] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
    max_stock_weight: float = MAX_STOCK_WEIGHT,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
    min_stocks: int = 10,
) -> Dict[str, float]:
    """
    Enforce all institutional constraints.
    
    Args:
        stock_weights: Initial weights
        shariah_compliant_only: Filter to Shariah-compliant only
        shariah_list: Set of compliant tickers
        sector_mapping: Ticker to sector mapping
        max_stock_weight: Maximum single stock weight
        max_sector_weight: Maximum sector weight
        min_stocks: Minimum number of stocks
    
    Returns:
        Constrained weights
    """
    from constraints.shariah_filter import enforce_shariah_portfolio
    
    weights = stock_weights.copy()
    
    # 1. Shariah filter
    if shariah_compliant_only:
        weights = enforce_shariah_portfolio(weights, shariah_list)
    
    # 2. Single stock limits
    weights = enforce_stock_limits(weights, max_stock_weight)
    
    # 3. Sector limits
    weights = enforce_sector_limits(weights, sector_mapping)
    
    # 4. Diversification check
    is_diversified, num_positions = check_diversification(weights, min_stocks)
    
    if not is_diversified:
        print(f"Warning: Only {num_positions} positions (min: {min_stocks})")
    
    return weights


if __name__ == "__main__":
    print("Sector Constraints Test")
    print("=" * 50)
    
    # Test portfolio
    test_portfolio = {
        "1155": 0.15,  # Maybank - Financials
        "1295": 0.12,  # PBBank - Financials
        "1023": 0.10,  # CIMB - Financials
        "6012": 0.08,  # Maxis - Telecom
        "5347": 0.08,  # Tenaga - Utilities
        "0166": 0.07,  # Inari - Technology
        "1961": 0.06,  # IOICORP - Plantations
        "3182": 0.05,  # Genting - Consumer Disc
        "6683": 0.05,  # IHH - Healthcare
        "0097": 0.04,  # Vitrox - Technology
    }
    
    # Calculate sector weights
    sector_weights = calculate_sector_weights(test_portfolio)
    
    print("\nInitial Sector Weights:")
    for sector, weight in sorted(sector_weights.items(), key=lambda x: -x[1]):
        limit = get_sector_limit(sector)
        status = "OK" if weight <= limit else "OVERWEIGHT"
        print(f"  {sector}: {weight:.1%} (limit: {limit:.0%}) - {status}")
    
    # Enforce constraints
    constrained = enforce_all_constraints(test_portfolio)
    
    print("\nConstrained Portfolio:")
    for ticker, weight in sorted(constrained.items(), key=lambda x: -x[1]):
        sector = get_sector(ticker)
        print(f"  {ticker} ({sector}): {weight:.2%}")
    
    # Final sector weights
    final_sectors = calculate_sector_weights(constrained)
    
    print("\nFinal Sector Weights:")
    for sector, weight in sorted(final_sectors.items(), key=lambda x: -x[1]):
        limit = get_sector_limit(sector)
        print(f"  {sector}: {weight:.1%} (limit: {limit:.0%})")
