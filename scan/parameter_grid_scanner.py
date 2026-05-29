"""
Parameter Grid Scanner for RSI Snap-Back Strategy.
Generates 1000+ configuration variants and runs 45-day rolling window backtests.
"""

import itertools
import json
import pandas as pd
import numpy as np
from decimal import Decimal
from pathlib import Path
from datetime import timedelta
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.persistence.wranglers import BarDataWrangler

from strategies.rsi_snap_back_long import RSISnapBackLong, RSISnapBackLongConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "btc_klines_15m_aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "scan_results"
GRID_SIZE = 300  # 2*2*2*2*3*2*2*2*2 = 768 combos
WINDOW_DAYS = 45
STEP_DAYS = 999  # single window


# ── Parameter Grid ──────────────────────────────────────────────────────────────
# Each list defines the values to sweep for one parameter.
# Total combinations = product of all list lengths.
PARAM_GRID = {
    # RSI indicator
    "rsi_period": [14, 20],

    # Stochastic indicator
    "stoch_period_k": [14, 20],
    "stoch_period_d": [3, 5],

    # EMA trend filter
    "ema_filter_period": [100, 200],

    # Entry thresholds
    "rsi_buy_threshold": [0.15, 0.20, 0.30],
    "stoch_buy_threshold": [20.0, 30.0],

    # Risk management
    "stop_loss_pct": [0.020, 0.030],
    "take_profit_pct": [0.060, 0.100],

    # Leverage
    "leverage": [3, 5],
}

# Fixed parameters
FIXED_PARAMS = {
    "trade_percent": Decimal("10"),
    "instrument_id_str": "BTCUSDT-PERP.BINANCE",
    "bar_type_str": "BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
}


def generate_param_grid() -> list[dict[str, Any]]:
    """Generate Cartesian product of PARAM_GRID."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    grid = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        # Add fixed params
        param_dict.update(FIXED_PARAMS)
        grid.append(param_dict)

    # sample evenly
    if len(grid) > GRID_SIZE:
        step = len(grid) / GRID_SIZE
        indices = sorted(set(int(i * step + 0.5) for i in range(GRID_SIZE)))
        grid = [grid[i] for i in indices if i < len(grid)]

    return grid


# ── Backtest helpers (must be top-level for multiprocessing) ───────────────────
def _engine_for_params(params: dict) -> BacktestEngine:
    """Create a BacktestEngine configured once per process."""
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("SCAN-001"),
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(10000, USDT)],
        default_leverage=Decimal(str(params.get("leverage", 5))),
    )
    return engine


def _extract_trades(engine, bar_type) -> list[dict]:
    """Extract completed round-trip trades from engine cache."""
    orders = engine.cache.orders()
    filled = sorted(
        [o for o in orders if o.filled_qty > 0 and o.ts_accepted is not None],
        key=lambda o: o.ts_accepted,
    )

    trades = []
    i = 0
    while i < len(filled) - 1:
        buy, sell = filled[i], filled[i + 1]
        if buy.side != 1:   # BUY == 1
            i += 1
            continue
        if sell.side != 2:  # SELL == 2
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

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": round(entry_px, 1),
            "exit_price": round(exit_px, 1),
            "quantity": round(qty, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": round(max(holding_days, 0), 4),
        })
        i += 2
    return trades


def _compute_metrics(trades_df: pd.DataFrame, initial_balance: float = 10000) -> dict:
    """Compute performance metrics from a trades dataframe."""
    if trades_df.empty:
        return {
            "net_return_pct": 0, "sharpe": 0, "max_drawdown": 0,
            "win_rate": 0, "profit_factor": 0, "sortino": 0,
            "total_trades": 0, "avg_holding_days": 0,
        }

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    win_rate = len(wins) / len(trades_df) * 100
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    total_pnl = trades_df["pnl"].sum()
    net_return_pct = (total_pnl / initial_balance) * 100

    cum_balance = initial_balance + trades_df["pnl"].cumsum()
    running_max = cum_balance.cummax()
    drawdown = (cum_balance - running_max) / running_max * 100
    max_dd = abs(drawdown.min())

    trade_returns = trades_df["pnl_pct"].values
    avg_r = trade_returns.mean()
    std_r = trade_returns.std()
    sharpe = avg_r / std_r * np.sqrt(365) if std_r > 0 else 0

    neg_r = trade_returns[trade_returns < 0]
    downside_std = neg_r.std() if len(neg_r) > 0 else 0.001
    sortino = avg_r / downside_std * np.sqrt(365)

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


def _run_single_config(args: tuple) -> dict:
    """Run 45-day rolling window backtest for one parameter config."""
    run_id, params, bars_data, window_days, step_days = args
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce, TriggerType

    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str(params["bar_type_str"])

    # Parse sweepable params
    sweepable = {
        k: v for k, v in params.items()
        if k in PARAM_GRID
    }
    fixed = {k: v for k, v in params.items() if k not in PARAM_GRID}

    results = []
    current = bars_data["start"]
    end = bars_data["end"]

    while current + timedelta(days=window_days) <= end:
        win_start = current
        win_end = current + timedelta(days=window_days)

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
            default_leverage=Decimal(str(sweepable.get("leverage", 5))),
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

        trades = _extract_trades(engine, bt)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = _compute_metrics(trades_df)
        metrics["window_start"] = str(win_start)
        metrics["window_end"] = str(win_end)
        results.append(metrics)

        current += timedelta(days=step_days)

    # Aggregate across windows
    if not results:
        return {"run_id": run_id, **sweepable, "status": "no_trades", "windows": 0}

    result_df = pd.DataFrame(results)
    agg = {
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
    return agg


# ── Main scanner ────────────────────────────────────────────────────────────────
def load_bars_data() -> dict:
    """Load raw DataFrame and wrap into nautilus bars."""
    df = pd.read_parquet(DATA_PATH).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    bars = wrangler.process(df)

    return {
        "bars": bars,
        "start": df.index.min(),
        "end": df.index.max(),
    }


def run_parameter_scan(n_workers: int = 1) -> pd.DataFrame:
    """
    Run parameter grid scan across all 1000+ config variants.
    Returns a DataFrame sorted by net_return_pct_avg descending.
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    bars_data = load_bars_data()
    print(f"Data range: {bars_data['start']} → {bars_data['end']}")

    print("Generating parameter grid...")
    param_grid = generate_param_grid()
    print(f"Total configurations: {len(param_grid)}")

    window_days = 45
    step_days = 45

    # Prepare args for workers
    scan_args = [
        (run_id, params, bars_data, window_days, step_days)
        for run_id, params in enumerate(param_grid)
    ]

    print(f"Running scan with {n_workers} workers...")
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_config, args): args[0] for args in scan_args}
        done = 0
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result(timeout=120)
                results.append(result)
                done += 1
                if done % 10 == 0:
                    print(f"  Completed: {done}/{len(scan_args)}")
            except Exception as e:
                print(f"  Run {run_id} failed: {e}")

    print(f"Scan complete. {len(results)}/{len(scan_args)} succeeded.")

    # Build DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("net_return_pct_avg", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)

    # Save full results
    out_path = OUTPUT_DIR / "scan_results_all.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Full results saved → {out_path}")

    # Save top 100
    top_path = OUTPUT_DIR / "scan_results_top100.csv"
    df_results.head(100).to_csv(top_path, index=False)
    print(f"Top 100 saved → {top_path}")

    return df_results


def print_summary(df_results: pd.DataFrame):
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
            f"leverage={int(row['leverage'])}  "
            f"rsi_th={row['rsi_buy_threshold']}  "
            f"stoch_th={row['stoch_buy_threshold']}  "
            f"sl={row['stop_loss_pct']}  tp={row['take_profit_pct']}  "
            f"rsi_p={int(row['rsi_period'])}  "
            f"stoch_k={int(row['stoch_period_k'])}  "
            f"stoch_d={int(row['stoch_period_d'])}  "
            f"ema={int(row['ema_filter_period'])}"
        )


if __name__ == "__main__":
    df = run_parameter_scan()
    print_summary(df)
    print("\nDone!")