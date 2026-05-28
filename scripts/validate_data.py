import duckdb
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET_PATH = DATA_DIR / "btc_klines_15m_aligned.parquet"

con = duckdb.connect()

con.execute(f"""
    CREATE TABLE btc_klines AS SELECT * FROM read_parquet('{PARQUET_PATH}')
""")

result = con.execute("""
    SELECT
        COUNT(*) AS total_bars,
        MIN(open_time) AS first_bar,
        MAX(open_time) AS last_bar,
        COUNT(DISTINCT DATE(open_time)) AS trading_days,
        ROUND(AVG(close), 2) AS avg_close,
        MIN(low) AS min_price,
        MAX(high) AS max_price
    FROM btc_klines
""").fetchdf()

print("=== DuckDB 数据验证 ===")
print(f"总 Bar 数:        {result['total_bars'][0]:,}")
print(f"数据范围:         {result['first_bar'][0]} -> {result['last_bar'][0]}")
print(f"交易日数:         {result['trading_days'][0]}")
print(f"平均收盘价:       ${result['avg_close'][0]:.2f}")
print(f"最低价:           ${result['min_price'][0]:.2f}")
print(f"最高价:           ${result['max_price'][0]:.2f}")

bars_per_year = con.execute("""
    SELECT
        YEAR(open_time) AS year,
        COUNT(*) AS bars
    FROM btc_klines
    GROUP BY year
    ORDER BY year
""").fetchdf()

print("\n=== 每年 Bar 数 ===")
for _, row in bars_per_year.iterrows():
    expected = 2976 * (12 if row['year'] not in (2023, 2026) else (8 if row['year'] == 2023 else 4))
    print(f"  {int(row['year'])}: {row['bars']:,} bars (预期 ≈ {expected:,})")

con.close()
