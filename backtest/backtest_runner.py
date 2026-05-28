import pandas as pd
import numpy as np
from decimal import Decimal
from pathlib import Path
from datetime import timedelta
from typing import Any

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


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def create_engine() -> BacktestEngine:
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(10000, USDT)],
        default_leverage=Decimal("5"),
    )
    return engine


def extract_trades(engine, start, end) -> list[dict[str, Any]]:
    orders = engine.cache.orders()
    filled = sorted(
        [o for o in orders if o.filled_qty > 0 and o.ts_accepted is not None],
        key=lambda o: o.ts_accepted,
    )

    trades = []
    i = 0
    while i < len(filled) - 1:
        buy = filled[i]
        sell = filled[i + 1]
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

        exit_type = {1: "MARKET", 2: "LIMIT(TP)", 3: "STOP_MARKET(SL)"}.get(sell.order_type, f"TYPE{sell.order_type}")

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": round(entry_px, 1),
            "exit_price": round(exit_px, 1),
            "quantity": qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": round(holding_days, 4),
            "reasoning": exit_type,
        })
        i += 2
    return trades


def run_backtest_full(start, end) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = load_data()
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    bars = wrangler.process(df)

    engine = create_engine()
    engine.add_instrument(inst)
    engine.add_data(bars)
    engine.add_strategy(RSISnapBackLong(
        RSISnapBackLongConfig(instrument_id=inst.id, bar_type=bt, trade_percent=10),
    ))
    engine.run(start=start, end=end)

    trades = extract_trades(engine, start, end)

    account = engine.cache.accounts()[0]
    initial_balance = 10000.0
    final_balance = float(account.balance(USDT).total.as_double())
    total_return = (final_balance - initial_balance) / initial_balance * 100

    return trades, {
        "initial_balance": initial_balance,
        "final_balance": round(final_balance, 2),
        "total_return": round(total_return, 2),
        "total_trades": len(trades),
    }


def compute_metrics(trades_df: pd.DataFrame, initial_balance: float = 10000) -> dict[str, Any]:
    if len(trades_df) == 0:
        return {
            "net_return_pct": 0, "sharpe": 0, "max_drawdown": 0,
            "win_rate": 0, "profit_factor": 0, "sortino": 0,
            "total_trades": 0, "avg_holding_days": 0,
        }

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    win_rate = len(wins) / len(trades_df) * 100
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = losses["pnl"].sum() if len(losses) > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    total_pnl = trades_df["pnl"].sum()
    net_return_pct = (total_pnl / initial_balance) * 100

    cum_balance = initial_balance + trades_df["pnl"].cumsum()
    running_max = cum_balance.cummax()
    drawdown = (cum_balance - running_max) / running_max * 100
    max_dd = abs(drawdown.min())

    holding_days = trades_df["holding_days"].clip(lower=0)
    avg_holding = holding_days.mean()

    trade_returns = trades_df["pnl_pct"].values
    avg_r = trade_returns.mean()
    std_r = trade_returns.std()
    sharpe = avg_r / std_r * np.sqrt(365) if std_r > 0 else 0

    neg_r = trade_returns[trade_returns < 0]
    downside_std = neg_r.std() if len(neg_r) > 0 else 0.001
    sortino = avg_r / downside_std * np.sqrt(365) if downside_std > 0 else 0

    return {
        "net_return_pct": round(net_return_pct, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sortino": round(sortino, 2),
        "total_trades": len(trades_df),
        "avg_holding_days": round(avg_holding, 1),
    }


def run_rolling_windows(data_df, window_days=45, step_days=45):
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    full_bars = wrangler.process(data_df)

    start = data_df.index.min()
    end = data_df.index.max()

    results = []
    current = start
    while current + timedelta(days=window_days) <= end:
        win_start = current
        win_end = current + timedelta(days=window_days)

        engine = create_engine()
        engine.add_instrument(inst)
        engine.add_data(full_bars)
        engine.add_strategy(RSISnapBackLong(
            RSISnapBackLongConfig(instrument_id=inst.id, bar_type=bt, trade_percent=10),
        ))
        engine.run(start=win_start, end=win_end)

        trades = extract_trades(engine, win_start, win_end)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = compute_metrics(trades_df)
        metrics["window_start"] = win_start
        metrics["window_end"] = win_end
        results.append(metrics)

        current += timedelta(days=step_days)

    return results


def run_all():
    df = load_data()
    full_start = df.index.min()
    full_end = df.index.max()

    print(f"Running 3-year full backtest: {full_start} -> {full_end}")
    trades, summary = run_backtest_full(full_start, full_end)
    trades_df = pd.DataFrame(trades)

    print(f"\n=== 3-Year Full Backtest Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if len(trades_df) > 0:
        metrics = compute_metrics(trades_df, summary["initial_balance"])
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print(f"\nRunning 45-day rolling window backtest...")
    rolling_results = run_rolling_windows(df)

    print(f"\n=== Rolling Windows Summary ===")
    print(f"Windows: {len(rolling_results)}")
    if rolling_results:
        for k in ["net_return_pct", "sharpe", "max_drawdown", "win_rate", "total_trades"]:
            values = [r[k] for r in rolling_results]
            print(f"  Avg {k}: {np.mean(values):.2f} (min={np.min(values):.2f}, max={np.max(values):.2f})")

    return trades_df, summary, rolling_results


if __name__ == "__main__":
    trades_df, summary, rolling_results = run_all()
    print("\nDone!")
