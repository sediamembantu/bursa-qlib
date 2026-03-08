"""
Liquidity Constraints Module

Ensures portfolio positions are in sufficiently liquid stocks.

Liquidity criteria:
- Minimum daily turnover (RM millions)
- Minimum market cap
- Bid-ask spread limits
- Trading frequency

Malaysian institutional guidelines:
- Minimum daily turnover: RM 1 million
- Minimum market cap: RM 500 million
- Maximum bid-ask spread: 2%
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_DAILY_TURNOVER


# =============================================================================
# Liquidity Thresholds
# =============================================================================

# Minimum daily turnover in RM millions
MIN_TURNOVER_RM = 1.0

# Minimum market cap in RM millions
MIN_MARKET_CAP_RM = 500.0

# Maximum bid-ask spread (%)
MAX_SPREAD_PCT = 2.0

# Minimum trading days in last 30 days
MIN_TRADING_DAYS = 20


# =============================================================================
# Liquidity Screening
# =============================================================================

def calculate_avg_turnover(
    price_df: pd.DataFrame,
    window: int = 30,
) -> float:
    """
    Calculate average daily turnover.
    
    Args:
        price_df: DataFrame with close and volume columns
        window: Rolling window in days
    
    Returns:
        Average daily turnover in RM
    """
    if price_df.empty:
        return 0.0
    
    df = price_df.tail(window).copy()
    
    # Calculate daily turnover
    df["turnover"] = df["close"] * df["volume"]
    
    return df["turnover"].mean()


def check_liquidity(
    ticker: str,
    price_df: pd.DataFrame,
    min_turnover: float = MIN_TURNOVER_RM,
    window: int = 30,
) -> Tuple[bool, str]:
    """
    Check if a stock meets liquidity requirements.
    
    Args:
        ticker: Stock code
        price_df: DataFrame with price and volume data
        min_turnover: Minimum average daily turnover (RM millions)
        window: Lookback window in days
    
    Returns:
        Tuple of (is_liquid, reason)
    """
    if price_df.empty:
        return False, "No price data"
    
    # Check trading frequency
    recent = price_df.tail(window)
    trading_days = len(recent[recent["volume"] > 0])
    
    if trading_days < MIN_TRADING_DAYS:
        return False, f"Only {trading_days} trading days in last {window}"
    
    # Check turnover
    avg_turnover = calculate_avg_turnover(price_df, window)
    avg_turnover_rm = avg_turnover / 1_000_000  # Convert to millions
    
    if avg_turnover_rm < min_turnover:
        return False, f"Avg turnover RM {avg_turnover_rm:.2f}M < {min_turnover}M"
    
    return True, "Liquid"


def screen_for_liquidity(
    tickers: List[str],
    price_data: Dict[str, pd.DataFrame],
    min_turnover: float = MIN_TURNOVER_RM,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Screen multiple stocks for liquidity.
    
    Args:
        tickers: List of ticker codes
        price_data: Dictionary of ticker -> price DataFrame
        min_turnover: Minimum daily turnover
    
    Returns:
        Tuple of (liquid_tickers, reasons)
    """
    liquid = []
    reasons = {}
    
    for ticker in tickers:
        price_df = price_data.get(ticker, pd.DataFrame())
        is_liquid, reason = check_liquidity(ticker, price_df, min_turnover)
        
        if is_liquid:
            liquid.append(ticker)
        else:
            reasons[ticker] = reason
    
    return liquid, reasons


# =============================================================================
# Position Sizing Based on Liquidity
# =============================================================================

def calculate_max_position(
    ticker: str,
    price_df: pd.DataFrame,
    portfolio_value: float,
    days_to_liquidate: int = 5,
    participation_rate: float = 0.10,
) -> float:
    """
    Calculate maximum position size based on liquidity.
    
    Args:
        ticker: Stock code
        price_df: Price DataFrame
        portfolio_value: Total portfolio value
        days_to_liquidate: Target days to exit position
        participation_rate: Max % of daily volume we'll trade
    
    Returns:
        Maximum position value in RM
    """
    avg_turnover = calculate_avg_turnover(price_df, window=30)
    
    # Maximum daily trading volume
    max_daily_trade = avg_turnover * participation_rate
    
    # Maximum position we can liquidate in target days
    max_position = max_daily_trade * days_to_liquidate
    
    # Also cap as % of portfolio
    portfolio_limit = portfolio_value * 0.10  # 10% max
    
    return min(max_position, portfolio_limit)


def adjust_for_liquidity(
    target_weights: Dict[str, float],
    price_data: Dict[str, pd.DataFrame],
    portfolio_value: float,
) -> Dict[str, float]:
    """
    Adjust portfolio weights based on liquidity constraints.
    
    Args:
        target_weights: Target ticker -> weight
        price_data: Dictionary of ticker -> price DataFrame
        portfolio_value: Total portfolio value
    
    Returns:
        Adjusted weights
    """
    adjusted = {}
    
    for ticker, weight in target_weights.items():
        price_df = price_data.get(ticker, pd.DataFrame())
        
        if price_df.empty:
            continue
        
        # Calculate liquidity-based max
        max_position = calculate_max_position(ticker, price_df, portfolio_value)
        max_weight = max_position / portfolio_value
        
        # Use minimum of target and liquidity limit
        adjusted[ticker] = min(weight, max_weight)
    
    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}
    
    return adjusted


# =============================================================================
# Liquidity-Weighted Portfolio
# =============================================================================

def liquidity_weighted_portfolio(
    predictions: Dict[str, float],
    price_data: Dict[str, pd.DataFrame],
    top_n: int = 20,
    min_turnover: float = MIN_TURNOVER_RM,
) -> Dict[str, float]:
    """
    Build portfolio weighted by both prediction and liquidity.
    
    Higher liquidity = higher weight capacity.
    
    Args:
        predictions: Ticker -> prediction score
        price_data: Ticker -> price DataFrame
        top_n: Number of stocks to consider
        min_turnover: Minimum turnover threshold
    
    Returns:
        Dictionary of ticker -> weight
    """
    # Filter liquid stocks
    liquid_tickers, _ = screen_for_liquidity(
        list(predictions.keys()),
        price_data,
        min_turnover,
    )
    
    if not liquid_tickers:
        return {}
    
    # Rank by prediction
    ranked = sorted(
        [(t, predictions[t]) for t in liquid_tickers],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    
    # Calculate liquidity scores
    liquidity_scores = {}
    for ticker, _ in ranked:
        price_df = price_data.get(ticker, pd.DataFrame())
        if not price_df.empty:
            avg_turnover = calculate_avg_turnover(price_df)
            liquidity_scores[ticker] = avg_turnover
    
    # Normalize liquidity scores
    total_liquidity = sum(liquidity_scores.values())
    if total_liquidity > 0:
        liquidity_scores = {t: s / total_liquidity for t, s in liquidity_scores.items()}
    
    # Combine prediction rank and liquidity
    weights = {}
    for i, (ticker, pred) in enumerate(ranked):
        # Prediction weight (higher rank = higher weight)
        pred_weight = (top_n - i) / top_n
        
        # Liquidity weight
        liq_weight = liquidity_scores.get(ticker, 1 / top_n)
        
        # Combined (50/50)
        weights[ticker] = 0.5 * pred_weight + 0.5 * liq_weight
    
    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    
    return weights


if __name__ == "__main__":
    print("Liquidity Constraints Test")
    print("=" * 50)
    
    # Load price data
    from pathlib import Path
    import pandas as pd
    
    price_dir = Path("data/raw/prices")
    price_data = {}
    
    for csv_file in list(price_dir.glob("*.csv"))[:5]:  # Test with 5 stocks
        ticker = csv_file.stem
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    
    print(f"\nLoaded {len(price_data)} stocks")
    
    # Test liquidity screening
    tickers = list(price_data.keys())
    liquid, reasons = screen_for_liquidity(tickers, price_data)
    
    print(f"\nLiquid stocks: {len(liquid)} / {len(tickers)}")
    print("\nIlliquid stocks:")
    for ticker, reason in reasons.items():
        print(f"  {ticker}: {reason}")
    
    # Test max position calculation
    print("\nMax Position Sizes (RM 1M portfolio):")
    for ticker in liquid[:3]:
        max_pos = calculate_max_position(ticker, price_data[ticker], 1_000_000)
        print(f"  {ticker}: RM {max_pos:,.0f}")
