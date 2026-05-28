#!/usr/bin/env python3
"""
BTC Snap-Back Long Strategy Backtest Runner
Runs 3-year full backtest + 45-day rolling windows, generates HTML report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.backtest_runner import run_backtest_full, run_rolling_windows, load_data, compute_metrics
from reports.report_generator import generate_report
import pandas as pd


def main():
    print("Loading data...")
    df = load_data()
    full_start = df.index.min()
    full_end = df.index.max()
    print(f"Data: {full_start} → {full_end} ({len(df)} bars)")

    print("Running 3-year full backtest...")
    trades, summary = run_backtest_full(full_start, full_end)
    trades_df = pd.DataFrame(trades)
    print(f"Trades: {len(trades_df)}, Final: ${summary['final_balance']:,.2f} (+{summary['total_return']:+.2f}%)")

    print("Running 45-day rolling windows...")
    rolling_results = run_rolling_windows(df)

    print("Generating HTML report...")
    path = generate_report(trades_df, summary, rolling_results)
    print(f"Done! Report: {path}")


if __name__ == "__main__":
    main()
