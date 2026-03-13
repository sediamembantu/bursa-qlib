#!/usr/bin/env python3
"""
Script 06: Convert to qlib-style Format

Converts existing CSV price data to the repository's qlib-style parquet
layout used by the local variant-B workflow.

Usage:
    python scripts/06_qlib_convert.py [--universe klci30]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.qlib.handler import BursaDataHandler
from config import QLIB_DIR
from tickers import get_all_tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV prices to qlib-style parquet")
    parser.add_argument("--universe", default="klci30", help="Universe to convert")
    args = parser.parse_args()

    tickers = get_all_tickers(args.universe)
    handler = BursaDataHandler()
    output_dir = QLIB_DIR / "features"

    print("=" * 60)
    print("CONVERT TO QLIB-STYLE FORMAT")
    print("=" * 60)
    print()
    print(f"Universe: {args.universe} ({len(tickers)} tickers)")
    print(f"Output:   {output_dir}")
    print()

    success_count = 0
    for ticker in tickers:
        output_path = handler.export_ticker(ticker, output_dir)
        if output_path is None:
            print(f"  [warn] {ticker}: no price file")
            continue

        print(f"  [ok]   {ticker}: {output_path.name}")
        success_count += 1

    instruments_path = handler.write_instruments_file(tickers)

    print()
    print("=" * 60)
    print(f"Converted {success_count}/{len(tickers)} tickers")
    print(f"Instruments file: {instruments_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
