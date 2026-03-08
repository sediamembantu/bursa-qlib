"""
Velocity Anomaly Detection

Detects rapid price movements (velocity) that may indicate:
- Breaking news
- Unusual market activity
- Potential front-running
- Technical breakouts/breakdowns

Velocity = Rate of change in price over short window
Acceleration = Rate of change in velocity
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Velocity Calculation
# =============================================================================

def calculate_price_velocity(
    price_df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Calculate price velocity (rate of change).
    
    Velocity = (Price_t - Price_{t-n}) / n
    
    Args:
        price_df: DataFrame with close column
        window: Window for velocity calculation
    
    Returns:
        DataFrame with velocity columns
    """
    df = price_df.copy()
    
    # Price velocity (absolute)
    df["price_velocity"] = (df["close"] - df["close"].shift(window)) / window
    
    # Price velocity (percentage)
    df["price_velocity_pct"] = (
        (df["close"] - df["close"].shift(window)) / 
        df["close"].shift(window) / window
    )
    
    # Acceleration (change in velocity)
    df["price_acceleration"] = df["price_velocity"] - df["price_velocity"].shift(window)
    
    return df


def calculate_volume_velocity(
    price_df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Calculate volume velocity (rate of change).
    
    Args:
        price_df: DataFrame with volume column
        window: Window for velocity calculation
    
    Returns:
        DataFrame with volume velocity columns
    """
    df = price_df.copy()
    
    # Volume velocity
    df["volume_velocity"] = (df["volume"] - df["volume"].shift(window)) / window
    
    # Volume velocity (percentage)
    df["volume_velocity_pct"] = (
        (df["volume"] - df["volume"].shift(window)) / 
        df["volume"].shift(window).replace(0, np.nan) / window
    )
    
    return df


# =============================================================================
# Velocity Anomaly Detection
# =============================================================================

def detect_velocity_anomalies(
    price_df: pd.DataFrame,
    velocity_window: int = 5,
    threshold_pct: float = 0.05,  # 5% per day
    acceleration_threshold: float = 0.02,  # 2% per day^2
) -> pd.DataFrame:
    """
    Detect velocity-based anomalies.
    
    Flags:
    - High positive velocity (rapid rise)
    - High negative velocity (rapid fall)
    - High acceleration (trend strengthening)
    
    Args:
        price_df: DataFrame with OHLCV data
        velocity_window: Window for velocity calculation
        threshold_pct: Velocity threshold (% per day)
        acceleration_threshold: Acceleration threshold
    
    Returns:
        DataFrame with velocity anomaly flags
    """
    df = price_df.copy()
    
    # Calculate velocity
    df = calculate_price_velocity(df, velocity_window)
    df = calculate_volume_velocity(df, velocity_window)
    
    # Price velocity anomalies
    df["velocity_anomaly_up"] = (
        df["price_velocity_pct"] > threshold_pct
    ).astype(int)
    
    df["velocity_anomaly_down"] = (
        df["price_velocity_pct"] < -threshold_pct
    ).astype(int)
    
    df["velocity_anomaly"] = (
        (df["velocity_anomaly_up"] == 1) | 
        (df["velocity_anomaly_down"] == 1)
    ).astype(int)
    
    # Acceleration anomalies
    df["acceleration_anomaly"] = (
        df["price_acceleration"].abs() > acceleration_threshold
    ).astype(int)
    
    # Combined velocity score
    df["velocity_score"] = (
        df["velocity_anomaly"] * 2 +
        df["acceleration_anomaly"] * 1
    )
    
    return df


# =============================================================================
# Trend Exhaustion Detection
# =============================================================================

def detect_trend_exhaustion(
    price_df: pd.DataFrame,
    rsi_window: int = 14,
    velocity_window: int = 5,
) -> pd.DataFrame:
    """
    Detect potential trend exhaustion using velocity + RSI divergence.
    
    Signals:
    - Overbought + decelerating = potential reversal down
    - Oversold + decelerating = potential reversal up
    
    Args:
        price_df: DataFrame with OHLCV data
        rsi_window: RSI calculation window
        velocity_window: Velocity window
    
    Returns:
        DataFrame with exhaustion signals
    """
    df = price_df.copy()
    
    # Calculate RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
    
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Calculate velocity
    df = calculate_price_velocity(df, velocity_window)
    
    # Overbought/Oversold
    df["overbought"] = (df["rsi"] > 70).astype(int)
    df["oversold"] = (df["rsi"] < 30).astype(int)
    
    # Deceleration
    df["decelerating"] = (
        (df["price_velocity_pct"].abs() < df["price_velocity_pct"].shift(5).abs()) &
        (df["price_velocity_pct"] != 0)
    ).astype(int)
    
    # Exhaustion signals
    df["exhaustion_top"] = (
        (df["overbought"] == 1) & 
        (df["decelerating"] == 1) &
        (df["price_velocity_pct"] > 0)
    ).astype(int)
    
    df["exhaustion_bottom"] = (
        (df["oversold"] == 1) & 
        (df["decelerating"] == 1) &
        (df["price_velocity_pct"] < 0)
    ).astype(int)
    
    return df


# =============================================================================
# Velocity Scan
# =============================================================================

def scan_velocity_anomalies(
    price_data: Dict[str, pd.DataFrame],
    velocity_window: int = 5,
    threshold_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Scan universe for velocity anomalies.
    
    Args:
        price_data: Dictionary of ticker -> price DataFrame
        velocity_window: Window for velocity calculation
        threshold_pct: Velocity threshold
    
    Returns:
        Summary DataFrame
    """
    results = []
    
    for ticker, df in price_data.items():
        # Detect velocity anomalies
        velocity_df = detect_velocity_anomalies(df, velocity_window, threshold_pct)
        
        # Get latest
        latest = velocity_df.iloc[-1]
        
        results.append({
            "ticker": ticker,
            "date": latest["date"],
            "close": latest["close"],
            "velocity_pct": latest.get("price_velocity_pct", 0) * 100,
            "velocity_anomaly": latest["velocity_anomaly"],
            "direction": "up" if latest.get("velocity_anomaly_up", 0) == 1 else 
                        "down" if latest.get("velocity_anomaly_down", 0) == 1 else "none",
            "acceleration": latest.get("price_acceleration", 0),
            "velocity_score": latest.get("velocity_score", 0),
        })
    
    df = pd.DataFrame(results)
    
    # Sort by velocity score
    df = df.sort_values("velocity_score", ascending=False)
    
    return df


if __name__ == "__main__":
    print("Velocity Anomaly Detection Test")
    print("=" * 50)
    
    # Load test data
    from pathlib import Path
    
    price_dir = Path("data/raw/prices")
    price_data = {}
    
    for csv_file in list(price_dir.glob("*.csv"))[:5]:
        ticker = csv_file.stem
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    
    print(f"\nLoaded {len(price_data)} stocks")
    
    # Scan for velocity anomalies
    summary = scan_velocity_anomalies(price_data)
    
    print("\nVelocity Summary:")
    print(summary[["ticker", "velocity_pct", "velocity_anomaly", "direction"]].to_string(index=False))
    
    # Test trend exhaustion
    print("\nTrend Exhaustion Test:")
    ticker = list(price_data.keys())[0]
    df = price_data[ticker]
    exhaustion_df = detect_trend_exhaustion(df)
    
    latest = exhaustion_df.iloc[-1]
    print(f"  {ticker}: RSI={latest['rsi']:.1f}, exhaustion_top={latest['exhaustion_top']}, exhaustion_bottom={latest['exhaustion_bottom']}")
