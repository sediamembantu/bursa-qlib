#!/usr/bin/env python3
"""
Script 05: Anomaly Scan

Runs the unified anomaly scanner and generates alerts.

Usage:
    python scripts/05_anomaly_scan.py [--min-score 25]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly.scanner import run_unified_scan, generate_alerts, format_alert_report
import pandas as pd
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run anomaly scan")
    parser.add_argument("--min-score", type=float, default=25.0, help="Minimum score for alert")
    parser.add_argument("--output", type=str, default="data/anomaly_reports", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BURSA-QLIB ANOMALY SCAN")
    print("Script 05: Anomaly Detection")
    print("=" * 60)
    
    # Load price data
    price_dir = Path("data/raw/prices")
    price_data = {}
    
    for csv_file in price_dir.glob("*.csv"):
        ticker = csv_file.stem
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    
    print(f"\nLoaded {len(price_data)} stocks")
    
    # Run scan
    results = run_unified_scan(price_data)
    
    # Generate alerts
    alerts = generate_alerts(results, min_score=args.min_score)
    
    # Print report
    report = format_alert_report(alerts)
    print("\n" + report)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    results_path = output_dir / f"scan_{timestamp}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total scanned: {len(results)}")
    print(f"Alerts: {len(alerts)}")
    
    if len(alerts) > 0:
        print(f"\nTop alerts:")
        for alert in alerts[:5]:
            print(f"  {alert['ticker']}: score={alert['score']:.0f}, priority={alert['priority']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
