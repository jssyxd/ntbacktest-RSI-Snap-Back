import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO


def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def draw_equity_curve(trades_df: pd.DataFrame, initial_balance: float) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    balances = [initial_balance]
    for _, t in trades_df.iterrows():
        balances.append(balances[-1] + t["pnl"])
    dates = [trades_df.iloc[0]["entry_time"]] + list(trades_df["exit_time"])

    ax.plot(dates, balances, color="#89b4fa", linewidth=1.5, label="Equity")
    ax.fill_between(dates, balances, alpha=0.1, color="#89b4fa")
    ax.axhline(y=initial_balance, color="#f38ba8", linestyle="--", linewidth=1, alpha=0.6, label=f"Initial (${initial_balance:,.0f})")

    ax.set_xlabel("Date", color="#cdd6f4")
    ax.set_ylabel("Balance (USDT)", color="#cdd6f4")
    ax.set_title("Equity Curve", color="#cdd6f4", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#cdd6f4")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")
    ax.grid(alpha=0.15, color="#585b70")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()

    return _fig_to_b64(fig)


def draw_drawdown(trades_df: pd.DataFrame, initial_balance: float) -> str:
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    balances = [initial_balance]
    for _, t in trades_df.iterrows():
        balances.append(balances[-1] + t["pnl"])
    dates = [trades_df.iloc[0]["entry_time"]] + list(trades_df["exit_time"])

    balances_series = pd.Series(balances, index=dates)
    running_max = balances_series.cummax()
    dd = (balances_series - running_max) / running_max * 100

    ax.fill_between(dates, dd, 0, color="#f38ba8", alpha=0.5, step="post")
    ax.fill_between(dates, dd, 0, where=(dd < -20), color="#f38ba8", alpha=0.8, step="post")
    ax.set_ylabel("Drawdown %", color="#cdd6f4")
    ax.set_title("Drawdown", color="#cdd6f4", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#cdd6f4")
    ax.grid(alpha=0.15, color="#585b70")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    y_min = dd.min() - 5
    ax.set_ylim(y_min, 2)
    fig.tight_layout()
    return _fig_to_b64(fig)


def draw_pnl_distribution(trades_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    pnl = trades_df["pnl_pct"]
    colors = ["#f38ba8" if v < 0 else "#a6e3a1" for v in pnl]
    ax.bar(range(len(pnl)), pnl, color=colors, width=0.7)

    ax.axhline(y=0, color="#585b70", linewidth=0.8)
    ax.set_xlabel("Trade #", color="#cdd6f4")
    ax.set_ylabel("PnL %", color="#cdd6f4")
    ax.set_title("Trade PnL Distribution", color="#cdd6f4", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#cdd6f4")
    ax.grid(alpha=0.15, color="#585b70", axis="y")
    fig.tight_layout()
    return _fig_to_b64(fig)


def draw_exit_type_pie(trades_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1e1e2e")

    counts = trades_df["reasoning"].value_counts()
    labels_map = {"STOP_SL": "Stop Loss (-2.5%)", "LIMIT_TP": "Take Profit (+7.5%)", "MARKET": "Market Close"}
    labels = [labels_map.get(k, k) for k in counts.index]
    colors_pie = ["#f38ba8" if "Stop" in k else "#a6e3a1" if "Take" in k else "#f9e2af" for k in labels]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors_pie,
        textprops={"color": "#cdd6f4", "fontsize": 10},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_color("#1e1e2e")
        t.set_fontweight("bold")
    ax.set_title("Exit Reason", color="#cdd6f4", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def draw_monthly_returns(trades_df: pd.DataFrame) -> str:
    df = trades_df.copy()
    df["month"] = df["exit_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    months = monthly.index.astype(str)
    values = monthly.values

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    colors_bar = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in values]
    ax.bar(range(len(months)), values, color=colors_bar, width=0.7)

    ax.axhline(y=0, color="#585b70", linewidth=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8, color="#cdd6f4")
    ax.set_ylabel("PnL (USDT)", color="#cdd6f4")
    ax.set_title("Monthly Returns", color="#cdd6f4", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#cdd6f4")
    ax.grid(alpha=0.15, color="#585b70", axis="y")
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_report(trades_df: pd.DataFrame, summary: dict, rolling_results: list[dict] | None = None, output_path: str | Path = "reports/backtest_report.html"):
    initial_balance = summary["initial_balance"]
    final_balance = summary["final_balance"]
    total_return = summary["total_return"]
    total_trades = summary["total_trades"]

    ewins = trades_df[trades_df["pnl"] > 0]
    elosses = trades_df[trades_df["pnl"] < 0]
    win_rate = len(ewins) / max(len(trades_df), 1) * 100
    avg_win = ewins["pnl_pct"].mean() if len(ewins) > 0 else 0
    avg_loss = elosses["pnl_pct"].mean() if len(elosses) > 0 else 0
    gross_profit = ewins["pnl"].sum() if len(ewins) > 0 else 0
    gross_loss = abs(elosses["pnl"].sum()) if len(elosses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    balances = [initial_balance]
    for _, t in trades_df.iterrows():
        balances.append(balances[-1] + t["pnl"])
    bs = pd.Series(balances)
    running_max = bs.cummax()
    dd = (bs - running_max) / running_max * 100
    max_dd = abs(dd.min())

    trade_returns = trades_df["pnl_pct"].values
    sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(365) if trade_returns.std() > 0 else 0
    neg_r = trade_returns[trade_returns < 0]
    downside_std = neg_r.std() if len(neg_r) > 0 else 0.001
    sortino = trade_returns.mean() / downside_std * np.sqrt(365) if downside_std > 0 else 0

    avg_holding_h = trades_df["holding_days"].mean() * 24 if len(trades_df) > 0 else 0
    best_trade = trades_df.loc[trades_df["pnl"].idxmax()] if len(trades_df) > 0 else None
    worst_trade = trades_df.loc[trades_df["pnl"].idxmin()] if len(trades_df) > 0 else None

    equity_img = draw_equity_curve(trades_df, initial_balance)
    dd_img = draw_drawdown(trades_df, initial_balance)
    pnl_dist_img = draw_pnl_distribution(trades_df)
    exit_pie_img = draw_exit_type_pie(trades_df)
    monthly_img = draw_monthly_returns(trades_df)

    trades_html_rows = ""
    for _, t in trades_df.iterrows():
        color = "#a6e3a1" if t["pnl"] > 0 else "#f38ba8"
        trades_html_rows += f"""<tr>
            <td>{t["entry_time"]}</td><td>{t["exit_time"]}</td>
            <td>{t["entry_price"]:.1f}</td><td>{t["exit_price"]:.1f}</td>
            <td>{t["pnl"]:.2f}</td>
            <td style="color:{color};font-weight:bold">{t["pnl_pct"]:+.2f}%</td>
            <td>{t["holding_days"]*24:.1f}h</td>
            <td>{t["reasoning"]}</td>
        </tr>"""

    best_row = worst_row = ""
    if best_trade is not None:
        best_row = f"""<tr><td>{best_trade["entry_time"]}</td><td>{best_trade["exit_time"]}</td>
            <td>{best_trade["entry_price"]:.1f}</td><td>{best_trade["exit_price"]:.1f}</td>
            <td style="color:#a6e3a1;font-weight:bold">{best_trade["pnl"]:+.2f}</td>
            <td style="color:#a6e3a1;font-weight:bold">{best_trade["pnl_pct"]:+.2f}%</td>
            <td>{best_trade["reasoning"]}</td></tr>"""
    if worst_trade is not None:
        worst_row = f"""<tr><td>{worst_trade["entry_time"]}</td><td>{worst_trade["exit_time"]}</td>
            <td>{worst_trade["entry_price"]:.1f}</td><td>{worst_trade["exit_price"]:.1f}</td>
            <td style="color:#f38ba8;font-weight:bold">{worst_trade["pnl"]:+.2f}</td>
            <td style="color:#f38ba8;font-weight:bold">{worst_trade["pnl_pct"]:+.2f}%</td>
            <td>{worst_trade["reasoning"]}</td></tr>"""

    rolling_html = ""
    if rolling_results:
        rolling_html = """<h2 style="color:#cdd6f4;border-bottom:2px solid #45475a;padding-bottom:8px">滚动窗口回测 (45天)</h2>
        <table class="stats"><tr>
            <th>窗口</th><th>起止</th><th>收益%</th><th>Sharpe</th><th>最大回撤%</th><th>胜率%</th><th>交易数</th>
        </tr>"""
        for r in rolling_results:
            ws = r["window_start"].strftime("%m-%d") if hasattr(r["window_start"], "strftime") else str(r["window_start"])
            we = r["window_end"].strftime("%m-%d") if hasattr(r["window_end"], "strftime") else str(r["window_end"])
            rolling_html += (
                f'<tr><td>{ws}-{we}</td><td>{ws} ~ {we}</td>'
                f'<td>{r["net_return_pct"]:+.2f}</td><td>{r["sharpe"]:.2f}</td>'
                f'<td>{r["max_drawdown"]:.2f}</td><td>{r["win_rate"]:.1f}</td>'
                f'<td>{r["total_trades"]}</td></tr>'
            )
        rolling_html += "</table>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Snap-Back Long 策略回测报告</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#11111b; color:#cdd6f4; font-family:-apple-system,'Segoe UI',sans-serif; padding:24px; }}
    h1 {{ color:#cba6f7; font-size:28px; margin-bottom:4px; }}
    .subtitle {{ color:#6c7086; font-size:14px; margin-bottom:24px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }}
    .card {{ background:#1e1e2e; border-radius:10px; padding:16px; border:1px solid #313244; }}
    .card .label {{ color:#6c7086; font-size:12px; text-transform:uppercase; letter-spacing:1px; }}
    .card .value {{ font-size:24px; font-weight:bold; margin-top:4px; }}
    .green {{ color:#a6e3a1; }} .red {{ color:#f38ba8; }} .blue {{ color:#89b4fa; }} .yellow {{ color:#f9e2af; }} .pink {{ color:#f5c2e7; }}
    table {{ width:100%; border-collapse:collapse; background:#1e1e2e; border-radius:10px; overflow:hidden; margin-bottom:24px; font-size:13px; }}
    th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #313244; }}
    th {{ background:#181825; color:#89b4fa; font-weight:600; }}
    tr:hover {{ background:#313244; }}
    td {{ font-family:'SF Mono','Fira Code',monospace; font-size:12px; }}
    .chart {{ background:#1e1e2e; border-radius:10px; padding:16px; margin-bottom:24px; border:1px solid #313244; }}
    .chart img {{ width:100%; height:auto; display:block; }}
    .stats td {{ font-family:inherit; }}
    .tabs {{ display:flex; gap:8px; margin-bottom:16px; }}
    .tab {{ padding:8px 16px; background:#313244; border-radius:6px; cursor:pointer; color:#6c7086; font-size:13px; }}
    .tab.active {{ background:#45475a; color:#cdd6f4; }}
    ::-webkit-scrollbar {{ width:8px; }} ::-webkit-scrollbar-track {{ background:#1e1e2e; }} ::-webkit-scrollbar-thumb {{ background:#45475a; border-radius:4px; }}
</style>
</head>
<body>

<h1>BTC Snap-Back Long 策略回测报告</h1>
<p class="subtitle">RSI + Stochastic 超卖反弹 · 15分钟K线 · NautilusTrader · 生成时间: {pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")}</p>

<div class="grid">
    <div class="card"><div class="label">总收益</div><div class="value green">{total_return:+.2f}%</div></div>
    <div class="card"><div class="label">最终权益</div><div class="value blue">${final_balance:,.2f}</div></div>
    <div class="card"><div class="label">总交易数</div><div class="value">{total_trades}</div></div>
    <div class="card"><div class="label">胜率</div><div class="value">{win_rate:.1f}%</div></div>
    <div class="card"><div class="label">盈亏比</div><div class="value yellow">{profit_factor:.2f}</div></div>
    <div class="card"><div class="label">最大回撤</div><div class="value pink">{max_dd:.2f}%</div></div>
</div>

<div class="grid">
    <div class="card"><div class="label">Sharpe Ratio</div><div class="value blue">{sharpe:.2f}</div></div>
    <div class="card"><div class="label">Sortino Ratio</div><div class="value blue">{sortino:.2f}</div></div>
    <div class="card"><div class="label">平均持有</div><div class="value">{avg_holding_h:.1f}h</div></div>
    <div class="card"><div class="label">平均盈利</div><div class="value green">{avg_win:+.2f}%</div></div>
    <div class="card"><div class="label">平均亏损</div><div class="value red">{avg_loss:+.2f}%</div></div>
    <div class="card"><div class="label">RV调整</div><div class="value">{total_return/max(max_dd,0.01)/5:.1f}</div></div>
</div>

<div class="chart"><img src="data:image/png;base64,{equity_img}" alt="Equity Curve"></div>
<div class="chart"><img src="data:image/png;base64,{dd_img}" alt="Drawdown"></div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
<div class="chart"><img src="data:image/png;base64,{pnl_dist_img}" alt="PnL Distribution"></div>
<div class="chart"><img src="data:image/png;base64,{exit_pie_img}" alt="Exit Reason"></div>
</div>

<div class="chart"><img src="data:image/png;base64,{monthly_img}" alt="Monthly Returns"></div>

{rolling_html}

<h2 style="color:#cdd6f4;border-bottom:2px solid #45475a;padding-bottom:8px;margin-bottom:12px">最优 / 最差交易</h2>
<table><tr><th>入场</th><th>出场</th><th>入场价</th><th>出场价</th><th>PnL</th><th>PnL%</th><th>原因</th></tr>
{best_row}
{worst_row}
</table>

<h2 style="color:#cdd6f4;border-bottom:2px solid #45475a;padding-bottom:8px;margin-bottom:12px">策略参数</h2>
<table class="stats">
    <tr><td>时间框架</td><td>15分钟</td></tr>
    <tr><td>数据范围</td><td>2023-05-01 ~ 2026-04-30 (3年)</td></tr>
    <tr><td>入场条件</td><td>RSI(14) &lt; 20 且 Stochastic(14,3,3) %K &lt; 25 且 价格 &gt; EMA200 × 0.9</td></tr>
    <tr><td>止损</td><td>2.5% (固定比例)</td></tr>
    <tr><td>止盈</td><td>7.5% (固定比例，R:R = 3:1)</td></tr>
    <tr><td>杠杆</td><td>5x</td></tr>
    <tr><td>每笔仓位</td><td>10% 保证金</td></tr>
    <tr><td>初始资金</td><td>$10,000 USDT</td></tr>
</table>

<h2 style="color:#cdd6f4;border-bottom:2px solid #45475a;padding-bottom:8px;margin-bottom:12px">全部交易明细 ({len(trades_df)}笔)</h2>
<div style="max-height:500px;overflow-y:auto;border-radius:10px">
<table><thead><tr>
    <th>入场时间</th><th>出场时间</th><th>入场价</th><th>出场价</th><th>PnL(USDT)</th><th>PnL%</th><th>持有</th><th>出场原因</th>
</tr></thead><tbody>
{trades_html_rows}
</tbody></table>
</div>

<div style="text-align:center;color:#6c7086;font-size:12px;margin-top:32px;padding:16px">
    BTC Snap-Back Long 策略 · NautilusTrader 回测引擎 · {pd.Timestamp.now("UTC").strftime("%Y-%m-%d")}
</div>

</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"Report saved to {path.resolve()}")
    return str(path)
