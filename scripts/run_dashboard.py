#!/usr/bin/env python3
"""
Script to run the Bursa-Qlib Dashboard.

Usage:
    python scripts/run_dashboard.py
    
Then open http://localhost:8501 in your browser.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("BURSA-QLIB DASHBOARD")
    print("=" * 60)
    print()
    print("Starting Streamlit dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run streamlit
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

if __name__ == "__main__":
    main()
