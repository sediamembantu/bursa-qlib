"""
Regime-Conditioned Model Selection

Uses detected market regimes to select appropriate model parameters
or trading strategies.

Regime strategies:
- risk_on: Aggressive positioning, higher beta exposure
- risk_off: Defensive positioning, lower turnover
- crisis: Reduce exposure, raise cash
- recovery: Increase exposure, favor rate-sensitive sectors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from regime.hmm_detector import get_regime_for_date


# =============================================================================
# Regime-Based Strategy Parameters
# =============================================================================

REGIME_CONFIGS = {
    "risk_on": {
        "top_n_stocks": 15,          # More stocks
        "max_position_pct": 0.12,    # Higher concentration
        "rebalance_freq": 5,         # Weekly rebalance
        "min_prediction": 0.0,       # Lower threshold
        "leverage": 1.0,             # No leverage
        "cash_buffer": 0.05,         # 5% cash
    },
    "risk_off": {
        "top_n_stocks": 8,           # Fewer stocks
        "max_position_pct": 0.08,    # Lower concentration
        "rebalance_freq": 10,        # Bi-weekly rebalance
        "min_prediction": 0.005,     # Higher threshold
        "leverage": 1.0,
        "cash_buffer": 0.15,         # 15% cash
    },
    "crisis": {
        "top_n_stocks": 5,           # Concentrated
        "max_position_pct": 0.05,    # Very low concentration
        "rebalance_freq": 20,        # Monthly rebalance
        "min_prediction": 0.01,      # High threshold
        "leverage": 0.8,             # Deleveraged
        "cash_buffer": 0.30,         # 30% cash
    },
    "recovery": {
        "top_n_stocks": 12,
        "max_position_pct": 0.10,
        "rebalance_freq": 5,
        "min_prediction": 0.0,
        "leverage": 1.1,             # Slight leverage
        "cash_buffer": 0.05,
    },
}


def get_regime_config(regime_name: str) -> dict:
    """
    Get strategy configuration for a regime.
    
    Args:
        regime_name: Name of regime
    
    Returns:
        Dictionary of strategy parameters
    """
    return REGIME_CONFIGS.get(regime_name, REGIME_CONFIGS["risk_on"])


# =============================================================================
# Regime-Aware Stock Selection
# =============================================================================

def filter_by_regime(
    predictions: Dict[str, float],
    regime_name: str,
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Filter/rank stocks based on regime-specific criteria.
    
    Args:
        predictions: Dictionary of ticker -> prediction
        regime_name: Current regime
        sector_mapping: Optional ticker -> sector mapping
    
    Returns:
        Filtered predictions
    """
    config = get_regime_config(regime_name)
    min_pred = config.get("min_prediction", 0.0)
    
    # Filter by minimum prediction
    filtered = {t: p for t, p in predictions.items() if p >= min_pred}
    
    # Regime-specific sector tilts
    if regime_name == "crisis":
        # In crisis, prefer defensive sectors
        if sector_mapping:
            defensive_sectors = ["Consumer Staples", "Utilities", "Healthcare"]
            filtered = {
                t: p * 1.2 if sector_mapping.get(t) in defensive_sectors else p
                for t, p in filtered.items()
            }
    
    elif regime_name == "recovery":
        # In recovery, prefer rate-sensitive sectors
        if sector_mapping:
            rate_sensitive = ["Financials", "Real Estate", "Industrials"]
            filtered = {
                t: p * 1.2 if sector_mapping.get(t) in rate_sensitive else p
                for t, p in filtered.items()
            }
    
    return filtered


# =============================================================================
# Regime-Aware Position Sizing
# =============================================================================

def adjust_position_size(
    ticker: str,
    base_weight: float,
    regime_name: str,
    is_glc: bool = False,
    is_shariah: bool = True,
) -> float:
    """
    Adjust position size based on regime and stock characteristics.
    
    Args:
        ticker: Stock ticker
        base_weight: Base weight (0-1)
        regime_name: Current regime
        is_glc: Whether stock is a GLC
        is_shariah: Whether stock is Shariah-compliant
    
    Returns:
        Adjusted weight
    """
    config = get_regime_config(regime_name)
    max_weight = config.get("max_position_pct", 0.10)
    
    # Start with base weight
    weight = base_weight
    
    # Cap at max
    weight = min(weight, max_weight)
    
    # Regime-specific adjustments
    if regime_name == "crisis":
        # Prefer GLCs in crisis (government backing)
        if is_glc:
            weight *= 1.1
        else:
            weight *= 0.9
    
    elif regime_name == "risk_off":
        # Slight preference for Shariah-compliant (ethical, stable)
        if is_shariah:
            weight *= 1.05
    
    # Final cap
    weight = min(weight, max_weight)
    
    return weight


# =============================================================================
# Regime Backtest Integration
# =============================================================================

def run_regime_conditioned_backtest(
    predictions_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """
    Run backtest with regime-conditioned parameters.
    
    Args:
        predictions_df: DataFrame with date, ticker, prediction columns
        regime_df: DataFrame with regime labels (date index)
        price_df: DataFrame with price data
        initial_capital: Starting capital
    
    Returns:
        DataFrame with NAV history
    """
    TRANSACTION_COST = 0.0025
    
    capital = initial_capital
    cash = initial_capital
    positions = {}
    nav_history = []
    
    dates = sorted(predictions_df["date"].unique())
    
    for i, date in enumerate(dates):
        # Get regime for this date
        regime_idx, regime_name = get_regime_for_date(pd.Timestamp(date), regime_df)
        config = get_regime_config(regime_name)
        
        # Check if rebalance day
        rebalance_freq = config.get("rebalance_freq", 5)
        if i % rebalance_freq != 0:
            # Just update NAV
            prices = price_df[price_df["date"] == date].set_index("ticker")["close"].to_dict()
            holdings_value = sum(positions.get(t, 0) * prices.get(t, 0) for t in positions)
            nav = cash + holdings_value
            nav_history.append({"date": date, "nav": nav, "regime": regime_name})
            continue
        
        # Get predictions for this date
        day_preds = predictions_df[predictions_df["date"] == date]
        
        # Filter by regime
        pred_dict = dict(zip(day_preds["ticker"], day_preds["prediction"]))
        filtered_preds = filter_by_regime(pred_dict, regime_name)
        
        # Select top N
        top_n = config.get("top_n_stocks", 10)
        top_stocks = sorted(filtered_preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        target_stocks = set([s[0] for s in top_stocks])
        
        # Get prices
        prices = price_df[price_df["date"] == date].set_index("ticker")["close"].to_dict()
        
        # Calculate NAV
        holdings_value = sum(positions.get(t, 0) * prices.get(t, 0) for t in positions)
        nav = cash + holdings_value
        
        # Calculate target weights (equal weight adjusted by regime)
        base_weight = (1 - config.get("cash_buffer", 0.05)) / top_n
        
        # Sell positions not in target
        for ticker in list(positions.keys()):
            if ticker not in target_stocks:
                shares = positions[ticker]
                price = prices.get(ticker, 0)
                if price > 0:
                    trade_value = shares * price
                    cost = trade_value * TRANSACTION_COST
                    cash += trade_value - cost
                    del positions[ticker]
        
        # Buy/update target positions
        for ticker, pred in top_stocks:
            price = prices.get(ticker, 0)
            if price <= 0:
                continue
            
            # Adjust weight
            adj_weight = adjust_position_size(ticker, base_weight, regime_name)
            
            # Calculate shares
            target_value = nav * adj_weight
            target_shares = int(target_value / price / 100) * 100  # Round to lots
            
            current_shares = positions.get(ticker, 0)
            
            if target_shares > current_shares:
                # Buy
                buy_shares = target_shares - current_shares
                trade_value = buy_shares * price
                cost = trade_value * TRANSACTION_COST
                
                if cash >= trade_value + cost:
                    cash -= trade_value + cost
                    positions[ticker] = target_shares
            
            elif target_shares < current_shares:
                # Sell
                sell_shares = current_shares - target_shares
                trade_value = sell_shares * price
                cost = trade_value * TRANSACTION_COST
                cash += trade_value - cost
                positions[ticker] = target_shares
        
        # Record NAV
        holdings_value = sum(positions.get(t, 0) * prices.get(t, 0) for t in positions)
        nav = cash + holdings_value
        nav_history.append({"date": date, "nav": nav, "regime": regime_name})
    
    return pd.DataFrame(nav_history)


if __name__ == "__main__":
    print("Regime-Conditioned Model Selection")
    print("=" * 50)
    
    # Show regime configs
    for regime, config in REGIME_CONFIGS.items():
        print(f"\n{regime}:")
        print(f"  Top N stocks: {config['top_n_stocks']}")
        print(f"  Max position: {config['max_position_pct']*100:.0f}%")
        print(f"  Rebalance freq: {config['rebalance_freq']} days")
        print(f"  Cash buffer: {config['cash_buffer']*100:.0f}%")
