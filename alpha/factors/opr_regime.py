"""
OPR Regime Factor

Classifies periods by Overnight Policy Rate (OPR) regime:
- Hiking: OPR increasing
- Holding: OPR stable
- Cutting: OPR decreasing

Bank stocks and rate-sensitive sectors perform differently under
different OPR regimes.

Factor:
    opr_regime: +1 (hiking), 0 (holding), -1 (cutting)
    opr_change: Magnitude of OPR change in basis points
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime


# Historical OPR data (simplified - in production, load from BNM data)
# Format: (effective_date, rate)
HISTORICAL_OPR = [
    ("2020-01-01", 3.00),
    ("2020-03-03", 2.50),  # COVID cut
    ("2020-05-05", 2.00),  # Further COVID cut
    ("2022-05-10", 2.25),  # First hike post-COVID
    ("2022-07-05", 2.50),
    ("2022-09-07", 2.75),
    ("2022-11-02", 3.00),
    ("2023-01-18", 2.75),  # Hold
    ("2023-03-08", 2.75),
    ("2023-05-10", 3.00),
    ("2024-01-01", 3.00),  # Holding
    ("2025-01-01", 2.75),  # Cut
    ("2026-03-05", 2.75),  # Current
]


def load_opr_history(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load historical OPR data.
    
    Args:
        filepath: Path to OPR CSV file
    
    Returns:
        DataFrame with date and rate columns
    """
    if filepath and filepath.exists():
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "rate"]].sort_values("date")
    
    # Use hardcoded data
    df = pd.DataFrame(HISTORICAL_OPR, columns=["date", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def classify_opr_regime(
    current_rate: float,
    previous_rate: float,
    threshold_bp: float = 10,
) -> int:
    """
    Classify OPR regime based on rate change.
    
    Args:
        current_rate: Current OPR rate
        previous_rate: Previous OPR rate
        threshold_bp: Minimum change in basis points to classify
    
    Returns:
        +1 (hiking), 0 (holding), -1 (cutting)
    """
    change_bp = (current_rate - previous_rate) * 100
    
    if change_bp > threshold_bp:
        return 1  # Hiking
    elif change_bp < -threshold_bp:
        return -1  # Cutting
    else:
        return 0  # Holding


def get_opr_regime_for_date(
    date: datetime,
    opr_history: pd.DataFrame,
) -> tuple[float, int]:
    """
    Get OPR rate and regime for a specific date.
    
    Args:
        date: Date to look up
        opr_history: DataFrame of OPR history
    
    Returns:
        Tuple of (rate, regime)
    """
    # Find applicable OPR (most recent before date)
    mask = opr_history["date"] <= date
    applicable = opr_history[mask]
    
    if applicable.empty:
        return 3.0, 0  # Default
    
    current_rate = applicable.iloc[-1]["rate"]
    
    # Get previous rate
    if len(applicable) > 1:
        previous_rate = applicable.iloc[-2]["rate"]
        regime = classify_opr_regime(current_rate, previous_rate)
    else:
        regime = 0
    
    return current_rate, regime


def add_opr_factors(
    price_df: pd.DataFrame,
    opr_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add OPR regime factors to price DataFrame.
    
    Args:
        price_df: DataFrame with date column
        opr_history: DataFrame of OPR history
    
    Returns:
        DataFrame with added opr_rate and opr_regime columns
    """
    if opr_history is None:
        opr_history = load_opr_history()
    
    df = price_df.copy()
    
    # Apply OPR data to each date
    opr_data = df["date"].apply(
        lambda d: get_opr_regime_for_date(pd.to_datetime(d), opr_history)
    )
    
    df["opr_rate"] = opr_data.apply(lambda x: x[0])
    df["opr_regime"] = opr_data.apply(lambda x: x[1])
    
    # Add regime dummies
    df["opr_hiking"] = (df["opr_regime"] == 1).astype(int)
    df["opr_cutting"] = (df["opr_regime"] == -1).astype(int)
    df["opr_holding"] = (df["opr_regime"] == 0).astype(int)
    
    return df


if __name__ == "__main__":
    print("Testing OPR regime factors...")
    
    # Create sample data
    dates = pd.date_range("2020-01-01", "2026-03-01", freq="M")
    df = pd.DataFrame({"date": dates, "close": np.random.uniform(10, 12, len(dates))})
    
    result = add_opr_factors(df)
    
    print("\nOPR regime over time:")
    print(result[["date", "opr_rate", "opr_regime"]].to_string(index=False))
    
    # Regime distribution
    print("\nRegime distribution:")
    print(result["opr_regime"].value_counts())
