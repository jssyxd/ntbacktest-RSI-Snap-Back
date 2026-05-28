import requests
import zipfile
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARQUET_DIR = Path(__file__).parent.parent / "data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

def download_month(year, month):
    filename = f"BTCUSDT-15m-{year}-{month:02d}.zip"
    url = BASE_URL + filename
    local_path = RAW_DIR / filename
    if local_path.exists():
        return f"{filename} already exists"
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return f"{filename} downloaded ({len(resp.content) / 1024:.0f} KB)"
    except Exception as e:
        return f"{filename} failed: {e}"

def parse_monthly_zip(year, month):
    filename = f"BTCUSDT-15m-{year}-{month:02d}.zip"
    zip_path = RAW_DIR / filename
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = f"BTCUSDT-15m-{year}-{month:02d}.csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.drop(columns=["ignore"])
    first_ts = df["open_time"].iloc[0]
    if first_ts > 1e15:
        unit = "us"
    else:
        unit = "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], unit=unit, errors="coerce")
    df = df.dropna(subset=["open_time"])
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--start-month", type=int, default=5)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-month", type=int, default=5)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    months = []
    y, m = args.start_year, args.start_month
    while (y < args.end_year) or (y == args.end_year and m <= args.end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    print(f"Downloading {len(months)} months of BTCUSDT 15m data...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(download_month, y, m): (y, m) for y, m in months}
        for f in tqdm(as_completed(futures), total=len(futures)):
            tqdm.write(f.result())

    if args.download_only:
        exit(0)

    print("\nParsing and converting to Parquet...")
    dfs = []
    for y, m in tqdm(months):
        df = parse_monthly_zip(y, m)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No data found!")
        exit(1)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values("open_time", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    full_df.to_parquet(PARQUET_DIR / "btc_klines_15m_aligned.parquet", index=False)

    print(f"\nTotal bars: {len(full_df)}")
    print(f"Date range: {full_df['open_time'].min()} -> {full_df['open_time'].max()}")
    print(f"Saved to: {PARQUET_DIR / 'btc_klines_15m_aligned.parquet'}")
