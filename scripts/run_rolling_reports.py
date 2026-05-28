#!/usr/bin/env python3
"""
Rolling window backtest: generates a full HTML report for each 45-day window.
"""

import sys, time
from pathlib import Path
from datetime import timedelta
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
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
from reports.report_generator import generate_report


def extract_trades(engine):
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
        if buy.side != 1 or sell.side != 2:
            i += 1
            continue
        entry_ts = pd.Timestamp(buy.ts_accepted, unit="ns", tz="UTC")
        exit_ts = pd.Timestamp(sell.ts_closed, unit="ns", tz="UTC") if sell.ts_closed else entry_ts
        holding_days = (exit_ts - entry_ts).total_seconds() / 86400
        pnl = (float(sell.avg_px) - float(buy.avg_px)) * float(buy.filled_qty)
        pnl_pct = (float(sell.avg_px) / float(buy.avg_px) - 1) * 100
        exit_type = {1: "MARKET", 2: "LIMIT_TP", 3: "STOP_SL"}.get(sell.order_type, f"TYPE{sell.order_type}")
        trades.append({
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "entry_price": round(float(buy.avg_px), 1),
            "exit_price": round(float(sell.avg_px), 1),
            "quantity": float(buy.filled_qty),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": round(holding_days, 4),
            "reasoning": exit_type,
        })
        i += 2
    return trades


def main():
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    bt = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")

    df = pd.read_parquet("data/btc_klines_15m_aligned.parquet").copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    wrangler = BarDataWrangler(bar_type=bt, instrument=inst)
    full_bars = wrangler.process(df)

    start = df.index.min()
    end = df.index.max()

    window_days = 45
    step_days = 45

    windows = []
    current = start
    while current + timedelta(days=window_days) <= end:
        windows.append((current, current + timedelta(days=window_days)))
        current += timedelta(days=step_days)

    total = len(windows)
    print(f"Generating {total} rolling window reports...")

    summary_rows = []
    t0_total = time.time()

    for idx, (ws, we) in enumerate(windows, 1):
        tw = time.time()
        engine = BacktestEngine(BacktestEngineConfig(
            trader_id=TraderId(f"ROLL-{idx:03d}"),
            logging=LoggingConfig(log_level="ERROR"),
        ))
        engine.add_venue(
            venue=Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(10000, USDT)],
            default_leverage=Decimal("5"),
        )
        engine.add_instrument(inst)
        engine.add_data(full_bars)
        engine.add_strategy(RSISnapBackLong(
            RSISnapBackLongConfig(instrument_id=inst.id, bar_type=bt, trade_percent=10),
        ))
        engine.run(start=ws, end=we)

        trades = extract_trades(engine)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        account = engine.cache.accounts()[0]
        final_bal = float(account.balance(USDT).total.as_double())
        total_return = (final_bal - 10000) / 10000 * 100

        summary = {
            "initial_balance": 10000.0,
            "final_balance": round(final_bal, 2),
            "total_return": round(total_return, 2),
            "total_trades": len(trades),
        }

        out_path = Path("reports/rolling") / f"window_{idx:03d}.html"
        generate_report(trades_df, summary, output_path=out_path)

        el = time.time() - tw
        print(f"  [{idx}/{total}] {ws.date()} - {we.date()}  trades={len(trades):<4}  "
              f"return={total_return:+.2f}%  final=${final_bal:,.2f}  ({el:.1f}s)")

        summary_rows.append({
            "window": f"{ws.date()} - {we.date()}",
            "trades": len(trades),
            "return_pct": round(total_return, 2),
            "final_balance": round(final_bal, 2),
            "time_s": round(el, 1),
        })

    # Generate index page
    idx_rows = ""
    for idx_num, sr in enumerate(summary_rows, 1):
        color_cls = "green" if sr["return_pct"] > 0 else "red"
        idx_url = f"window_{idx_num:03d}.html"
        idx_rows += f'<tr><td><a href="{idx_url}">{sr["window"]}</a></td>' \
                    f'<td>{sr["trades"]}</td>' \
                    f'<td class="{color_cls}">{sr["return_pct"]:+.2f}%</td>' \
                    f'<td>${sr["final_balance"]:,.2f}</td>' \
                    f'<td>{sr["time_s"]}s</td></tr>\n'

    index_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>45天滚动窗口回测索引</title>
<style>
    body{{background:#11111b;color:#cdd6f4;font-family:-apple-system,sans-serif;padding:24px;max-width:960px;margin:0 auto}}
    h1{{color:#cba6f7;font-size:24px}}
    table{{width:100%;border-collapse:collapse;background:#1e1e2e;border-radius:10px;overflow:hidden}}
    th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #313244}}
    th{{background:#181825;color:#89b4fa}}
    tr:hover{{background:#313244}}
    a{{color:#89b4fa;text-decoration:none}}
    a:hover{{text-decoration:underline}}
    .green{{color:#a6e3a1}}.red{{color:#f38ba8}}
</style>
</head>
<body>
<h1>45天滚动窗口回测报告索引</h1>
<p style="color:#6c7086">共 {total} 个窗口 · NautilusTrader · {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M UTC')}</p>
<table><thead><tr>
<th>窗口</th><th>交易数</th><th>收益率</th><th>最终权益</th><th>耗时</th>
</tr></thead><tbody>
{idx_rows}
</tbody></table>
<p style="color:#6c7086;font-size:12px;margin-top:16px">
    <a href="../backtest_report.html">← 返回 3年全程报告</a>
</p>
</body></html>"""

    Path("reports/rolling/index.html").write_text(index_html, encoding="utf-8")

    print(f"\nTotal time: {time.time()-t0_total:.1f}s")
    print(f"Reports: reports/rolling/index.html")


if __name__ == "__main__":
    main()
