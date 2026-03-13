"""
Unified Anomaly Scanner

Combines all anomaly detection methods:
1. Z-score anomalies (price, volume)
2. Velocity anomalies (rapid movements)
3. KNN outliers (cross-sectional)

Produces unified alert score and priority ranking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly.zscore import detect_all_anomalies, scan_universe_anomalies
from anomaly.velocity import detect_velocity_anomalies, scan_velocity_anomalies
from anomaly.knn_detector import detect_cross_sectional_anomalies, build_feature_matrix


# =============================================================================
# Combined Anomaly Score
# =============================================================================

def calculate_combined_score(
    zscore_anomaly: int,
    velocity_anomaly: int,
    knn_outlier: int,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate combined anomaly score.
    
    Args:
        zscore_anomaly: Z-score anomaly flag (0 or 1)
        velocity_anomaly: Velocity anomaly flag (0 or 1)
        knn_outlier: KNN outlier flag (0 or 1)
        weights: Custom weights for each method
    
    Returns:
        Combined anomaly score (0-100)
    """
    if weights is None:
        weights = {
            "zscore": 0.4,      # Most reliable
            "velocity": 0.35,   # Good for short-term
            "knn": 0.25,        # Cross-sectional
        }
    
    score = (
        zscore_anomaly * weights["zscore"] +
        velocity_anomaly * weights["velocity"] +
        knn_outlier * weights["knn"]
    )
    
    return score * 100  # Scale to 0-100


# =============================================================================
# Unified Scanner
# =============================================================================

def run_unified_scan(
    price_data: Dict[str, pd.DataFrame],
    window: int = 20,
    velocity_threshold_pct: float = 0.05,
    knn_contamination: float = 0.1,
) -> pd.DataFrame:
    """
    Run all anomaly detection methods and combine results.
    
    Args:
        price_data: Dictionary of ticker -> price DataFrame
        window: Rolling window for calculations
        velocity_threshold_pct: Velocity threshold
        knn_contamination: Expected outlier proportion
    
    Returns:
        Combined anomaly report
    """
    print("Running unified anomaly scan...")
    
    # 1. Z-score scan
    print("  Scanning Z-score anomalies...")
    zscore_summary, zscore_details = scan_universe_anomalies(price_data, window)
    zscore_dict = dict(zip(zscore_summary["ticker"], zscore_summary["has_anomaly"]))
    
    # 2. Velocity scan
    print("  Scanning velocity anomalies...")
    velocity_summary = scan_velocity_anomalies(price_data, window, velocity_threshold_pct)
    velocity_dict = dict(zip(velocity_summary["ticker"], velocity_summary["velocity_anomaly"]))
    
    # 3. KNN scan
    print("  Scanning KNN outliers...")
    knn_results, knn_outliers = detect_cross_sectional_anomalies(
        price_data, window, k=5, contamination=knn_contamination
    )
    knn_dict = dict(zip(knn_results.index, knn_results["is_outlier"])) if not knn_results.empty else {}
    
    # Combine results
    all_tickers = set(zscore_dict.keys()) | set(velocity_dict.keys()) | set(knn_dict.keys())
    
    results = []
    for ticker in all_tickers:
        zscore_flag = zscore_dict.get(ticker, 0)
        velocity_flag = velocity_dict.get(ticker, 0)
        knn_flag = knn_dict.get(ticker, 0)
        
        # Get latest price info
        if ticker in price_data:
            latest = price_data[ticker].iloc[-1]
            close = latest["close"]
            date = latest["date"]
        else:
            close = 0
            date = None
        
        # Combined score
        combined_score = calculate_combined_score(zscore_flag, velocity_flag, knn_flag)
        
        results.append({
            "ticker": ticker,
            "date": date,
            "close": close,
            "zscore_anomaly": zscore_flag,
            "velocity_anomaly": velocity_flag,
            "knn_outlier": knn_flag,
            "combined_score": combined_score,
            "anomaly_count": zscore_flag + velocity_flag + knn_flag,
        })
    
    df = pd.DataFrame(results)
    
    # Sort by combined score
    df = df.sort_values("combined_score", ascending=False)
    
    # Add priority level
    df["priority"] = pd.cut(
        df["combined_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )
    
    return df


# =============================================================================
# Alert Generation
# =============================================================================

def generate_alerts(
    scan_results: pd.DataFrame,
    min_score: float = 25.0,
) -> List[Dict]:
    """
    Generate alerts for anomalies above threshold.
    
    Args:
        scan_results: Results from run_unified_scan
        min_score: Minimum combined score for alert
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    high_score = scan_results[scan_results["combined_score"] >= min_score]
    
    for _, row in high_score.iterrows():
        alert = {
            "ticker": row["ticker"],
            "date": str(row["date"]),
            "close": row["close"],
            "priority": row["priority"],
            "score": row["combined_score"],
            "signals": [],
        }
        
        if row["zscore_anomaly"] == 1:
            alert["signals"].append("zscore_anomaly")
        
        if row["velocity_anomaly"] == 1:
            alert["signals"].append("velocity_anomaly")
        
        if row["knn_outlier"] == 1:
            alert["signals"].append("cross_sectional_outlier")
        
        alerts.append(alert)
    
    return alerts


def format_alert_report(alerts: List[Dict]) -> str:
    """
    Format alerts as readable report.
    
    Args:
        alerts: List of alert dictionaries
    
    Returns:
        Formatted string
    """
    if not alerts:
        return "No anomalies detected."
    
    lines = [
        "=" * 60,
        f"ANOMALY ALERT REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"\nTotal alerts: {len(alerts)}",
        "",
    ]
    
    # Group by priority
    priorities = ["critical", "high", "medium", "low"]
    
    for priority in priorities:
        priority_alerts = [a for a in alerts if a["priority"] == priority]
        
        if priority_alerts:
            lines.append(f"\n{priority.upper()} PRIORITY ({len(priority_alerts)})")
            lines.append("-" * 40)
            
            for alert in priority_alerts:
                lines.append(f"\n  {alert['ticker']}")
                lines.append(f"    Score: {alert['score']:.0f}")
                lines.append(f"    Price: {alert['close']:.2f}")
                lines.append(f"    Signals: {', '.join(alert['signals'])}")
    
    return "\n".join(lines)


# =============================================================================
# Main Scanner Script
# =============================================================================

def main():
    """Run unified anomaly scanner."""
    print("=" * 60)
    print("BURSA-QLIB ANOMALY SCANNER")
    print("=" * 60)
    
    # Load price data
    from pathlib import Path
    
    price_dir = Path("data/raw/prices")
    price_data = {}
    
    for csv_file in price_dir.glob("*.csv"):
        ticker = csv_file.stem
        if ticker == "combined_prices":
            continue
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    
    print(f"\nLoaded {len(price_data)} stocks")
    
    # Run scan
    results = run_unified_scan(price_data)
    
    # Generate alerts
    alerts = generate_alerts(results, min_score=25.0)
    
    # Print report
    report = format_alert_report(alerts)
    print("\n" + report)
    
    # Save results
    output_dir = Path("data/anomaly_reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    results_path = output_dir / f"scan_{timestamp}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total stocks scanned: {len(results)}")
    print(f"Anomalies detected: {len(alerts)}")
    print(f"\nBy priority:")
    print(results["priority"].value_counts().to_string())
    
    if len(alerts) > 0:
        print(f"\nBy signal type:")
        signal_counts = {"zscore": 0, "velocity": 0, "knn": 0}
        for alert in alerts:
            for signal in alert["signals"]:
                if "zscore" in signal:
                    signal_counts["zscore"] += 1
                elif "velocity" in signal:
                    signal_counts["velocity"] += 1
                elif "cross_sectional" in signal:
                    signal_counts["knn"] += 1
        
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count}")


if __name__ == "__main__":
    main()
