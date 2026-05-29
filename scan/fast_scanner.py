"""
Fast Parameter Grid Scanner for RSI Snap-Back Strategy.
Single process, sequential execution, checkpointed.
"""

import gc
import itertools
import json
import time
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

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
WINDOW_DAYS = 45
STEP_DAYS = 9999  # single window only (end-start < step, so only 1 window)


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

GRID_SIZE = 1200  # 7*5*2*3*4*4*4*3*2 = 80640 combos, sample to 1200


def generate_param_grid():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    # Sample to GRID_SIZE evenly
    if len(grid) > GRID_SIZE:
        step = len(grid) / GRID_SIZE
        indices = sorted(set(int(i * step + 0.5) for i in range(GRID_SIZE)))
        grid = [grid[i] for i in indices if i < len(grid)]
    return grid


def load_bars_data():
    print("Loading data...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    bars = wrangler.process(df)

    start = df.index.min()
    end = df.index.max()
    print(f"done ({len(bars)} bars, {start} -> {end})")
    return {"bars": bars, "start": start, "end": end, "inst": inst, "bt": bt}


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"net_return_pct": 0, "sharpe": 0, "max_drawdown": 0,
                "win_rate": 0, "profit_factor": 0, "sortino": 0,
                "total_trades": 0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    win_rate = len(wins) / len(trades_df) * 100
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.001
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
    downside_std = neg_r.std() if len(neg_r) > 0 else 0.001
    sortino = avg_r / downside_std * 6.0
    return {
        "net_return_pct": round(net_return_pct, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sortino": round(sortino, 2),
        "total_trades": int(len(trades_df)),
    }


def extract_trades(engine):
    orders = engine.cache.orders()
    filled = sorted(
        [o for o in orders if o.filled_qty > 0 and o.ts_accepted is not None],
        key=lambda o: o.ts_accepted,
    )
    trades = []
    i = 0
    while i < len(filled) - 1:
        buy, sell = filled[i], filled[i + 1]
        if buy.side != 1:
            i += 1
            continue
        if sell.side != 2:
            i += 2
            continue
        entry_px = float(buy.avg_px)
        exit_px = float(sell.avg_px)
        qty = float(buy.filled_qty)
        entry_time = pd.Timestamp(buy.ts_accepted, unit="ns", tz="UTC")
        exit_time = pd.Timestamp(sell.ts_accepted, unit="ns", tz="UTC")
        holding_days = (exit_time - entry_time).total_seconds() / 86400
        pnl = (exit_px - entry_px) * qty
        pnl_pct = (exit_px / entry_px - 1) * 100
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                        "entry_price": entry_px, "exit_price": exit_px,
                        "quantity": qty, "pnl": pnl, "pnl_pct": pnl_pct,
                        "holding_days": holding_days})
        i += 2
    return trades


def run_single_config(run_id, sweepable, bars_data, inst, bt):
    current = bars_data["start"]
    end = bars_data["end"]

    results = []
    while current + timedelta(days=WINDOW_DAYS) <= end:
        win_start = current
        win_end = current + timedelta(days=WINDOW_DAYS)

        engine = BacktestEngine(
            BacktestEngineConfig(
                trader_id=TraderId(f"SCAN-{run_id:04d}"),
                logging=LoggingConfig(log_level="ERROR"),
            ),
        )
        engine.add_venue(
            venue=Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(10000, USDT)],
            default_leverage=Decimal(str(int(sweepable["leverage"]))),
        )
        engine.add_instrument(inst)
        engine.add_data(bars_data["bars"])
        engine.add_strategy(RSISnapBackLong(
            RSISnapBackLongConfig(
                instrument_id=inst.id,
                bar_type=bt,
                rsi_period=int(sweepable["rsi_period"]),
                stoch_period_k=int(sweepable["stoch_period_k"]),
                stoch_period_d=int(sweepable["stoch_period_d"]),
                ema_filter_period=int(sweepable["ema_filter_period"]),
                rsi_buy_threshold=float(sweepable["rsi_buy_threshold"]),
                stoch_buy_threshold=float(sweepable["stoch_buy_threshold"]),
                stop_loss_pct=float(sweepable["stop_loss_pct"]),
                take_profit_pct=float(sweepable["take_profit_pct"]),
                leverage=int(sweepable["leverage"]),
                trade_percent=Decimal("10"),
            ),
        ))
        engine.run(start=win_start, end=win_end)

        trades = extract_trades(engine)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = compute_metrics(trades_df)
        metrics["window_start"] = str(win_start)
        metrics["window_end"] = str(win_end)
        results.append(metrics)

        current += timedelta(days=STEP_DAYS)
        del engine
        gc.collect()

    if not results:
        return {"run_id": run_id, **sweepable, "status": "no_trades", "windows": 0}

    result_df = pd.DataFrame(results)
    return {
        "run_id": run_id,
        **sweepable,
        "status": "ok",
        "windows": len(results),
        "net_return_pct_avg": round(result_df["net_return_pct"].mean(), 2),
        "net_return_pct_min": round(result_df["net_return_pct"].min(), 2),
        "net_return_pct_max": round(result_df["net_return_pct"].max(), 2),
        "sharpe_avg": round(result_df["sharpe"].mean(), 2),
        "max_drawdown_avg": round(result_df["max_drawdown"].mean(), 2),
        "win_rate_avg": round(result_df["win_rate"].mean(), 1),
        "profit_factor_avg": round(result_df["profit_factor"].mean(), 2),
        "sortino_avg": round(result_df["sortino"].mean(), 2),
        "total_trades_sum": int(result_df["total_trades"].sum()),
        "total_trades_avg": round(result_df["total_trades"].mean(), 1),
    }


def run_scan():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    param_grid = generate_param_grid()
    print(f"Total configurations: {len(param_grid)}")

    bars_data = load_bars_data()
    inst = bars_data["inst"]
    bt = bars_data["bt"]

    results = []
    t0 = time.time()
    CHECKPOINT = 50  # save partial results every N configs
    for i, sweepable in enumerate(param_grid):
        result = run_single_config(i, sweepable, bars_data, inst, bt)
        results.append(result)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = len(param_grid) - i - 1
            eta = remaining / rate
            print(f"  Progress: {i+1}/{len(param_grid)}, elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s ({eta/60:.1f}min)")
            sys.stdout.flush()
        if (i + 1) % CHECKPOINT == 0:
            # Save checkpoint
            df_partial = pd.DataFrame(results)
            df_partial = df_partial.sort_values("net_return_pct_avg", ascending=False).reset_index(drop=True)
            df_partial["rank"] = range(1, len(df_partial) + 1)
            ckpt_path = OUTPUT_DIR / "scan_checkpoint.csv"
            df_partial.to_csv(ckpt_path, index=False)
            print(f"  [CHECKPOINT {i+1}] saved to {ckpt_path}")
            sys.stdout.flush()

    print(f"Scan complete. {len(results)}/{len(param_grid)} succeeded in {time.time()-t0:.0f}s")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("net_return_pct_avg", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)

    out_path = OUTPUT_DIR / "scan_results_all.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Full results saved → {out_path}")

    top_path = OUTPUT_DIR / "scan_results_top100.csv"
    df_results.head(min(100, len(df_results))).to_csv(top_path, index=False)
    print(f"Top results saved → {top_path}")

    # Print summary
    print("\n=== Parameter Scan Summary ===")
    print(f"Total configs tested: {len(df_results)}")
    print(f"Configs with trades:   {(df_results['status'] == 'ok').sum()}")

    top = df_results[df_results["status"] == "ok"].head(10)
    print(f"\nTop 10 by avg return:")
    for _, row in top.iterrows():
        print(
            f"  #{int(row['rank']):3d}  "
            f"ret_avg={row['net_return_pct_avg']:+.1f}%  "
            f"ret_min={row['net_return_pct_min']:+.1f}%  "
            f"ret_max={row['net_return_pct_max']:+.1f}%  "
            f"sharpe={row['sharpe_avg']:.2f}  "
            f"win={row['win_rate_avg']:.0f}%  "
            f"lev={int(row['leverage'])}  "
            f"rsi_th={row['rsi_buy_threshold']}  "
            f"stoch_th={row['stoch_buy_threshold']}  "
            f"sl={row['stop_loss_pct']}  tp={row['take_profit_pct']}  "
            f"rsi_p={int(row['rsi_period'])}  "
            f"stoch_k={int(row['stoch_period_k'])}  "
            f"stoch_d={int(row['stoch_period_d'])}  "
            f"ema={int(row['ema_filter_period'])}"
        )

    return df_results


if __name__ == "__main__":
    df = run_scan()
    print("\nDone!")