#!/usr/bin/env python3
"""
Script 08: Backtest qlib-style Variant

Backtests the local qlib-style model using the same Bursa trading assumptions
as Variant A.

Usage:
    python scripts/08_qlib_backtest.py [--model models/qlib_variant_xxx.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.qlib.handler import BursaDataHandler
from config import CLEARING_FEE_RATE, COMMISSION_RATE, LOT_SIZE, STAMP_DUTY_RATE
from constraints.shariah_filter import is_shariah_compliant
from tickers import get_all_tickers


def load_model(model_path: str) -> lgb.Booster:
    """Load a trained LightGBM model from disk."""
    return lgb.Booster(model_file=model_path)


def calculate_transaction_costs(trade_value: float, is_buy: bool = True) -> dict[str, float]:
    """Calculate Malaysian transaction costs for a single trade."""
    commission = max(trade_value * COMMISSION_RATE, 8)
    stamp_duty = min(trade_value * STAMP_DUTY_RATE, 200)
    clearing_fee = trade_value * CLEARING_FEE_RATE

    return {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "clearing_fee": clearing_fee,
        "total": commission + stamp_duty + clearing_fee,
    }


def adjust_for_lot_size(shares: int) -> int:
    """Round a share count down to the nearest Bursa lot."""
    return (shares // LOT_SIZE) * LOT_SIZE


def portfolio_value(cash: float, holdings: dict[str, int], prices: dict[str, float]) -> float:
    """Calculate current cash plus marked-to-market holdings."""
    holdings_value = sum(holdings.get(ticker, 0) * prices.get(ticker, 0) for ticker in holdings)
    return cash + holdings_value


def prepare_features(
    tickers: list[str],
    start_date: str,
    end_date: str,
    handler: BursaDataHandler | None = None,
) -> pd.DataFrame:
    """Load a qlib-style feature frame for backtesting."""
    if handler is None:
        handler = BursaDataHandler()

    combined, _ = handler.prepare_training_frame(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    return combined


def backtest_strategy(
    df: pd.DataFrame,
    model: lgb.Booster,
    initial_capital: float = 1_000_000,
    rebalance_freq: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    """Backtest a long-only equal-weight strategy with transaction costs."""
    feature_cols = model.feature_name()
    missing_cols = [column for column in feature_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Backtest data is missing model feature columns: {missing_cols}")

    print(f"Using {len(feature_cols)} features")
    print(f"Features: {feature_cols[:5]}...")

    df = df.copy()
    df["pred"] = model.predict(df[feature_cols])

    dates = df["date"].sort_values().unique()
    rebalance_dates = dates[::rebalance_freq]

    cash = initial_capital
    holdings: dict[str, int] = {}
    nav_history: list[dict[str, float | int | pd.Timestamp]] = []

    for date in dates:
        date_df = df[df["date"] == date].copy()
        prices = date_df.set_index("ticker")["close"].to_dict()

        if date in rebalance_dates:
            date_df["shariah"] = date_df["ticker"].apply(is_shariah_compliant)
            date_df = date_df[date_df["shariah"]].copy()

            if date_df.empty:
                continue

            selected = date_df.nlargest(top_n, "pred")
            current_nav = portfolio_value(cash, holdings, prices)
            target_value = current_nav / len(selected)
            target_holdings = {
                row["ticker"]: adjust_for_lot_size(int(target_value / row["close"]))
                for _, row in selected.iterrows()
            }

            for ticker in list(holdings.keys()):
                if ticker in target_holdings:
                    continue

                price = prices.get(ticker)
                shares = holdings.get(ticker, 0)
                if price is None or shares <= 0:
                    continue

                trade_value = shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=False)
                cash += trade_value - costs["total"]
                del holdings[ticker]

            for ticker, target_shares in target_holdings.items():
                current_shares = holdings.get(ticker, 0)
                if target_shares >= current_shares:
                    continue

                price = prices[ticker]
                sell_shares = current_shares - target_shares
                trade_value = sell_shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=False)
                cash += trade_value - costs["total"]
                holdings[ticker] = target_shares

            for _, row in selected.sort_values("pred", ascending=False).iterrows():
                ticker = row["ticker"]
                price = row["close"]
                current_shares = holdings.get(ticker, 0)
                target_shares = target_holdings[ticker]
                if target_shares <= current_shares:
                    continue

                buy_shares = target_shares - current_shares
                trade_value = buy_shares * price
                costs = calculate_transaction_costs(trade_value, is_buy=True)
                total_cost = trade_value + costs["total"]

                if total_cost > cash:
                    affordable_shares = adjust_for_lot_size(int(cash / price))
                    buy_shares = max(0, affordable_shares)
                    trade_value = buy_shares * price
                    costs = calculate_transaction_costs(trade_value, is_buy=True)
                    total_cost = trade_value + costs["total"]

                if buy_shares <= 0 or total_cost > cash:
                    continue

                cash -= total_cost
                holdings[ticker] = current_shares + buy_shares

        nav_history.append(
            {
                "date": date,
                "nav": portfolio_value(cash, holdings, prices),
                "n_holdings": len(holdings),
            }
        )

    return pd.DataFrame(nav_history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest qlib-style variant")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument("--universe", default="klci30", help="Universe to backtest")
    parser.add_argument("--start-date", default="2024-01-01", help="Backtest start date")
    parser.add_argument("--end-date", default="2025-12-31", help="Backtest end date")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Rebalance frequency in days")
    parser.add_argument("--top-n", type=int, default=10, help="Number of holdings")
    args = parser.parse_args()

    print("=" * 60)
    print("QLIB-STYLE VARIANT BACKTEST")
    print("=" * 60)
    print()

    if args.model:
        model_path = Path(args.model)
    else:
        models = list(Path("models").glob("qlib_variant_*.txt"))
        if not models:
            print("[error] No qlib-style model found")
            print("Run: python scripts/07_qlib_train.py")
            return
        model_path = max(models, key=lambda x: x.stat().st_mtime)

    print(f"Model: {model_path}")

    model = load_model(str(model_path))

    print("\nLoading data...")
    df = prepare_features(
        tickers=get_all_tickers(args.universe),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if df.empty:
        raise ValueError(
            "No backtest data available. Run scripts/01_fetch_data.py first to create price files."
        )

    print(f"Features: {len(df)} rows")

    print("\nRunning backtest...")
    nav_df = backtest_strategy(
        df,
        model,
        initial_capital=args.capital,
        rebalance_freq=args.rebalance_freq,
        top_n=args.top_n,
    )

    nav_df["return"] = nav_df["nav"].pct_change()
    total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1) * 100
    trading_days = len(nav_df)
    annual_return = total_return * (252 / trading_days) if trading_days > 0 else 0
    volatility = nav_df["return"].std() * np.sqrt(252) * 100 if len(nav_df) > 1 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0

    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Volatility: {volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Trading Days: {trading_days}")

    output_path = Path("backtest_results") / (
        f"qlib_nav_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(output_path, index=False)

    print(f"\n[ok] Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("COMPARISON vs VARIANT A")
    print("=" * 60)
    print("| Metric      | Variant A | Variant B |")
    print("|-------------|-----------|-----------|")
    print(f"| Return      | +4.74%    | {total_return:+.2f}%    |")
    print(f"| Sharpe      | 0.37      | {sharpe:.2f}      |")
    print(f"| Max DD      | -4.86%    | {max_drawdown:.2f}%    |")


if __name__ == "__main__":
    main()
