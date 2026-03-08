"""
Z-Score Anomaly Detection

Detects anomalies in price and volume using Z-score method.

Anomaly types:
- Price spikes: Large daily moves beyond normal range
- Volume surges: Unusual trading activity
- Volatility regime changes: Sudden increase in volatility
- Correlation breakdowns: Stocks moving independently

Z-score threshold: |z| > 3 indicates significant anomaly
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Z-Score Calculation
# =============================================================================

def calculate_z_score(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Calculate rolling Z-score.
    
    Z = (x - μ) / σ
    
    Args:
        series: Input series
        window: Rolling window for mean/std
    
    Returns:
        Series of Z-scores
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    
    z = (series - mean) / std
    
    return z


def detect_price_anomalies(
    price_df: pd.DataFrame,
    window: int = 20,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect price anomalies using Z-score.
    
    Args:
        price_df: DataFrame with close column
        window: Rolling window
        threshold: Z-score threshold for anomaly
    
    Returns:
        DataFrame with anomaly flags
    """
    df = price_df.copy()
    
    # Daily returns
    df["return"] = df["close"].pct_change()
    
    # Z-score of returns
    df["return_z"] = calculate_z_score(df["return"], window)
    
    # Anomaly flag
    df["price_anomaly"] = (df["return_z"].abs() > threshold).astype(int)
    
    # Direction
    df["anomaly_direction"] = np.where(
        df["price_anomaly"] == 1,
        np.where(df["return_z"] > 0, "up", "down"),
        "none"
    )
    
    return df


def detect_volume_anomalies(
    price_df: pd.DataFrame,
    window: int = 20,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect volume anomalies using Z-score.
    
    Args:
        price_df: DataFrame with volume column
        window: Rolling window
        threshold: Z-score threshold
    
    Returns:
        DataFrame with volume anomaly flags
    """
    df = price_df.copy()
    
    # Volume Z-score (use log to handle large variations)
    df["log_volume"] = np.log1p(df["volume"])
    df["volume_z"] = calculate_z_score(df["log_volume"], window)
    
    # Anomaly flag
    df["volume_anomaly"] = (df["volume_z"].abs() > threshold).astype(int)
    
    return df


def detect_volatility_regime_change(
    price_df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect volatility regime changes.
    
    Compares short-term vs long-term volatility.
    
    Args:
        price_df: DataFrame with close column
        short_window: Short-term volatility window
        long_window: Long-term volatility window
        threshold: Ratio threshold for regime change
    
    Returns:
        DataFrame with regime change flags
    """
    df = price_df.copy()
    
    # Daily returns
    df["return"] = df["close"].pct_change()
    
    # Short-term and long-term volatility
    df["vol_short"] = df["return"].rolling(short_window).std()
    df["vol_long"] = df["return"].rolling(long_window).std()
    
    # Ratio
    df["vol_ratio"] = df["vol_short"] / df["vol_long"]
    
    # Regime change flag
    df["vol_regime_change"] = (df["vol_ratio"] > threshold).astype(int)
    
    return df


# =============================================================================
# Combined Anomaly Detection
# =============================================================================

def detect_all_anomalies(
    price_df: pd.DataFrame,
    window: int = 20,
    price_threshold: float = 3.0,
    volume_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect all types of anomalies.
    
    Args:
        price_df: DataFrame with OHLCV data
        window: Rolling window
        price_threshold: Price Z-score threshold
        volume_threshold: Volume Z-score threshold
    
    Returns:
        DataFrame with all anomaly flags
    """
    df = price_df.copy()
    
    # Price anomalies
    df = detect_price_anomalies(df, window, price_threshold)
    
    # Volume anomalies
    df = detect_volume_anomalies(df, window, volume_threshold)
    
    # Volatility regime changes
    df = detect_volatility_regime_change(df)
    
    # Combined anomaly score
    df["anomaly_score"] = (
        df["price_anomaly"] * 3 +
        df["volume_anomaly"] * 2 +
        df["vol_regime_change"] * 1
    )
    
    # Any anomaly flag
    df["has_anomaly"] = (
        (df["price_anomaly"] == 1) |
        (df["volume_anomaly"] == 1) |
        (df["vol_regime_change"] == 1)
    ).astype(int)
    
    return df


# =============================================================================
# Universe-Level Anomaly Scan
# =============================================================================

def scan_universe_anomalies(
    price_data: Dict[str, pd.DataFrame],
    window: int = 20,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Scan entire universe for anomalies.
    
    Args:
        price_data: Dictionary of ticker -> price DataFrame
        window: Rolling window
        top_n: Number of top anomalies to return
    
    Returns:
        Tuple of (summary DataFrame, list of anomaly details)
    """
    results = []
    details = []
    
    for ticker, df in price_data.items():
        # Detect anomalies
        anomaly_df = detect_all_anomalies(df, window)
        
        # Get latest row
        latest = anomaly_df.iloc[-1]
        
        results.append({
            "ticker": ticker,
            "date": latest["date"],
            "close": latest["close"],
            "price_anomaly": latest["price_anomaly"],
            "anomaly_direction": latest["anomaly_direction"],
            "volume_anomaly": latest["volume_anomaly"],
            "vol_regime_change": latest["vol_regime_change"],
            "anomaly_score": latest["anomaly_score"],
            "has_anomaly": latest["has_anomaly"],
            "return_z": latest.get("return_z", 0),
            "volume_z": latest.get("volume_z", 0),
        })
        
        # Collect details for anomalies
        if latest["has_anomaly"] == 1:
            details.append({
                "ticker": ticker,
                "date": str(latest["date"]),
                "close": latest["close"],
                "price_anomaly": bool(latest["price_anomaly"]),
                "direction": latest["anomaly_direction"],
                "volume_anomaly": bool(latest["volume_anomaly"]),
                "vol_change": bool(latest["vol_regime_change"]),
                "return_z": round(latest.get("return_z", 0), 2),
                "volume_z": round(latest.get("volume_z", 0), 2),
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Sort by anomaly score
    summary_df = summary_df.sort_values("anomaly_score", ascending=False)
    
    return summary_df.head(top_n), details


if __name__ == "__main__":
    print("Z-Score Anomaly Detection Test")
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
    
    # Scan for anomalies
    summary, details = scan_universe_anomalies(price_data)
    
    print("\nAnomaly Summary (Top 5):")
    print(summary[["ticker", "price_anomaly", "volume_anomaly", "anomaly_score"]].to_string(index=False))
    
    if details:
        print(f"\nAnomaly Details ({len(details)} detected):")
        for d in details[:3]:
            print(f"  {d['ticker']}: price={d['price_anomaly']}, vol={d['volume_anomaly']}, ret_z={d['return_z']}")
