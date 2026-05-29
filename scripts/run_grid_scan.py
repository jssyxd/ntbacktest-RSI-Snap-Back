#!/usr/bin/env python3
"""
Parameter Grid Scanner for RSI Snap-Back Strategy.

Usage:
    python3 scripts/run_grid_scan.py [--configs N] [--workers N]

Outputs:
    scan_results/scan_results_all.csv   - All configs, sorted by net_return_pct_avg desc
    scan_results/scan_results_top100.csv - Top 100 configs
"""

import argparse
import gc
import itertools
import time
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strategies.rsi_snap_back_long import RSISnapBackLong, RSISnapBackLongConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "btc_klines_15m_aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "scan_results"

# ── Parameter Grid ──────────────────────────────────────────────────────────────
PARAM_GRID = {
    "rsi_period": [8, 10, 12, 14, 16, 20, 24],
    "stoch_period_k": [8, 10, 12, 14, 16],
    "stoch_period_d": [3, 5],
    "ema_filter_period": [100, 150, 200],
    "rsi_buy_threshold": [0.15, 0.20, 0.25, 0.30],
    "stoch_buy_threshold": [15.0, 20.0, 25.0, 30.0],
    "stop_loss_pct": [0.015, 0.020, 0.025, 0.030],
    "take_profit_pct": [0.050, 0.075, 0.100],
    "leverage": [3, 5],
}

WINDOW_DAYS = 45
STEP_DAYS = 9999  # single 45-day window (end-start < step → 1 window)


def generate_param_grid(target_size: int = 1200):
    """Cartesian product of PARAM_GRID, sampled evenly to target_size."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if len(grid) > target_size:
        step = len(grid) / target_size
        indices = sorted(
            set(int(i * step + 0.5) for i in range(target_size))
        )
        grid = [grid[i] for i in indices if i < len(grid)]

    return grid


def load_data():
    """Load parquet → nautilus bars."""
    print("Loading data...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    bars = wrangler.process(df)

    print(f"done ({len(bars)} bars, {df.index.min().date()} → {df.index.max().date()})")
    return {"bars": bars, "start": df.index.min(), "end": df.index.max(), "inst": inst, "bt": bt}


def compute_metrics(trades_df):
    """Compute performance metrics from a trades DataFrame."""
    if trades_df.empty:
        return {k: 0 for k in (
            "net_return_pct", "sharpe", "max_drawdown",
            "win_rate", "profit_factor", "sortino",
            "total_trades", "avg_holding_days",
        )}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    win_rate = len(wins) / len(trades_df) * 100
    gross_profit = wins["pnl"].sum() if len(wins) else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0.001
    profit_factor = gross_profit / gross_loss
    total_pnl = trades_df["pnl"].sum()
    net_return_pct = (total_pnl / 10000) * 100

    cum_balance = 10000 + trades_df["pnl"].cumsum()
    running_max = cum_balance.cummax()
    dd = (cum_balance - running_max) / running_max * 100
    max_dd = abs(dd.min())

    trade_returns = trades_df["pnl_pct"].values
    avg_r = trade_returns.mean()
    std_r = trade_returns.std()
    sharpe = avg_r / std_r * 6.0 if std_r > 0 else 0
    neg_r = trade_returns[trade_returns < 0]
    downside_std = neg_r.std() if len(neg_r) else 0.001
    sortino = avg_r / downside_std * 6.0

    return {
        "net_return_pct": round(net_return_pct, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sortino": round(sortino, 2),
        "total_trades": int(len(trades_df)),
        "avg_holding_days": round(trades_df["holding_days"].clip(lower=0).mean(), 1),
    }


def extract_trades(engine):
    """Extract completed BUY→SELL round-trip trades."""
    orders = sorted(
        [o for o in engine.cache.orders() if o.filled_qty > 0 and o.ts_accepted],
        key=lambda o: o.ts_accepted,
    )
    trades = []
    i = 0
    while i < len(orders) - 1:
        buy, sell = orders[i], orders[i + 1]
        if buy.side != 1 or sell.side != 2:
            i += 1
            continue
        entry_px = float(buy.avg_px)
        exit_px = float(sell.avg_px)
        qty = float(buy.filled_qty)
        entry_time = pd.Timestamp(buy.ts_accepted, unit="ns", tz="UTC")
        exit_time = pd.Timestamp(sell.ts_accepted, unit="ns", tz="UTC")
        holding_days = (exit_time - entry_time).total_seconds() / 86400
        pnl = (exit_px - entry_px) * qty
        pnl_pct = (exit_px / entry_px - 1) * 100
        trades.append({
            "entry_time": entry_time, "exit_time": exit_time,
            "entry_price": entry_px, "exit_price": exit_px,
            "quantity": qty, "pnl": pnl, "pnl_pct": pnl_pct,
            "holding_days": max(holding_days, 0),
        })
        i += 2
    return trades


def run_single_config(run_id, cfg, bars_data):
    """Run 45-day backtest for one parameter config."""
    inst, bt = bars_data["inst"], bars_data["bt"]
    current = bars_data["start"]
    end = bars_data["end"]

    results = []
    while current + timedelta(days=WINDOW_DAYS) <= end:
        win_end = current + timedelta(days=WINDOW_DAYS)

        engine = BacktestEngine(
            BacktestEngineConfig(
                trader_id=TraderId(f"SCAN-{run_id:04d}"),
                logging=LoggingConfig(log_level="ERROR"),
            ),
        )
        engine.add_venue(
            Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(10000, USDT)],
            default_leverage=Decimal(str(int(cfg["leverage"]))),
        )
        engine.add_instrument(inst)
        engine.add_data(bars_data["bars"])
        engine.add_strategy(RSISnapBackLong(
            RSISnapBackLongConfig(
                instrument_id=inst.id,
                bar_type=bt,
                rsi_period=int(cfg["rsi_period"]),
                stoch_period_k=int(cfg["stoch_period_k"]),
                stoch_period_d=int(cfg["stoch_period_d"]),
                ema_filter_period=int(cfg["ema_filter_period"]),
                rsi_buy_threshold=float(cfg["rsi_buy_threshold"]),
                stoch_buy_threshold=float(cfg["stoch_buy_threshold"]),
                stop_loss_pct=float(cfg["stop_loss_pct"]),
                take_profit_pct=float(cfg["take_profit_pct"]),
                leverage=int(cfg["leverage"]),
                trade_percent=Decimal("10"),
            ),
        ))
        engine.run(start=current, end=win_end)

        trades = extract_trades(engine)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = compute_metrics(trades_df)
        metrics["window_start"] = str(current)
        metrics["window_end"] = str(win_end)
        results.append(metrics)

        current += timedelta(days=STEP_DAYS)
        del engine
        gc.collect()

    if not results:
        return {"run_id": run_id, **cfg, "status": "no_trades", "windows": 0}

    df_w = pd.DataFrame(results)
    return {
        "run_id": run_id,
        **cfg,
        "status": "ok",
        "windows": len(results),
        "net_return_pct_avg": round(df_w["net_return_pct"].mean(), 2),
        "net_return_pct_min": round(df_w["net_return_pct"].min(), 2),
        "net_return_pct_max": round(df_w["net_return_pct"].max(), 2),
        "sharpe_avg": round(df_w["sharpe"].mean(), 2),
        "max_drawdown_avg": round(df_w["max_drawdown"].mean(), 2),
        "win_rate_avg": round(df_w["win_rate"].mean(), 1),
        "profit_factor_avg": round(df_w["profit_factor"].mean(), 2),
        "sortino_avg": round(df_w["sortino"].mean(), 2),
        "total_trades_sum": int(df_w["total_trades"].sum()),
        "total_trades_avg": round(df_w["total_trades"].mean(), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="RSI Snap-Back Parameter Grid Scanner")
    parser.add_argument("--configs", type=int, default=1200,
                        help="Target number of configs to sample (default: 1200)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    param_grid = generate_param_grid(target_size=args.configs)
    total_combos = 1
    for v in PARAM_GRID.values():
        total_combos *= len(v)
    print(f"Parameter grid: {total_combos:,} total combos → sampled to {len(param_grid)}")
    print(f"Dimensions: rsi_period={PARAM_GRID['rsi_period']}")
    print(f"            stoch_k={PARAM_GRID['stoch_period_k']}, stoch_d={PARAM_GRID['stoch_period_d']}")
    print(f"            ema={PARAM_GRID['ema_filter_period']}")
    print(f"            rsi_th={PARAM_GRID['rsi_buy_threshold']}")
    print(f"            stoch_th={PARAM_GRID['stoch_buy_threshold']}")
    print(f"            sl={PARAM_GRID['stop_loss_pct']}, tp={PARAM_GRID['take_profit_pct']}")
    print(f"            leverage={PARAM_GRID['leverage']}")
    print()

    bars_data = load_data()
    inst, bt = bars_data["inst"], bars_data["bt"]

    results = []
    t0 = time.time()
    for i, cfg in enumerate(param_grid):
        result = run_single_config(i, cfg, bars_data)
        results.append(result)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(param_grid) - i - 1) / rate
            print(f"  [{i+1}/{len(param_grid)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s ({eta/60:.1f}min)")
            sys.stdout.flush()

        if (i + 1) % 50 == 0:
            df_ckpt = pd.DataFrame(results).sort_values(
                "net_return_pct_avg", ascending=False
            ).reset_index(drop=True)
            df_ckpt["rank"] = range(1, len(df_ckpt) + 1)
            ckpt_path = OUTPUT_DIR / "scan_checkpoint.csv"
            df_ckpt.to_csv(ckpt_path, index=False)
            print(f"  [CHECKPOINT {i+1}] → {ckpt_path}")
            sys.stdout.flush()

    print(f"\nScan complete: {len(results)}/{len(param_grid)} configs in {time.time()-t0:.0f}s")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        "net_return_pct_avg", ascending=False
    ).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)

    all_path = OUTPUT_DIR / "scan_results_all.csv"
    df_results.to_csv(all_path, index=False)
    print(f"Full results → {all_path}  ({len(df_results)} rows)")

    top_path = OUTPUT_DIR / "scan_results_top100.csv"
    df_results.head(min(100, len(df_results))).to_csv(top_path, index=False)
    print(f"Top 100      → {top_path}")

    # Summary
    ok = df_results[df_results["status"] == "ok"]
    print(f"\n=== Summary ===")
    print(f"Total configs: {len(df_results)}")
    print(f"With trades:  {len(ok)}")
    print(f"\nTop 10 by net_return_pct_avg:")
    for _, row in ok.head(10).iterrows():
        print(
            f"  #{int(row['rank']):3d}  "
            f"ret={row['net_return_pct_avg']:+.2f}%  "
            f"sharpe={row['sharpe_avg']:.2f}  "
            f"win={row['win_rate_avg']:.0f}%  "
            f"dd={row['max_drawdown_avg']:.1f}%  "
            f"lev={int(row['leverage'])}  "
            f"rsi_th={row['rsi_buy_threshold']}  "
            f"stoch_th={row['stoch_buy_threshold']}  "
            f"sl={row['stop_loss_pct']}  tp={row['take_profit_pct']}  "
            f"rsi_p={int(row['rsi_period'])}  sk={int(row['stoch_period_k'])}  "
            f"sd={int(row['stoch_period_d'])}  ema={int(row['ema_filter_period'])}"
        )
    print("\nDone!")


if __name__ == "__main__":
    main()