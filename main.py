from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from config import load_config
from data.binance import BinanceClient
from data.database import MarketDatabase
from data.downloader import download_timeframe
from features.feature_engine import add_features, pattern_vector

console = Console()
HISTORICAL_DIR = Path("data/historical")
HISTORICAL_COLUMNS = [
    "symbol", "timeframe", "open_time", "close_time", "open", "high", "low",
    "close", "volume", "quote_volume", "trades", "is_closed",
]


def cmd_init() -> None:
    cfg = load_config()
    MarketDatabase(cfg.database).close()
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Database ready:[/green] {cfg.database}")


def cmd_download(candles: int | None) -> None:
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    client = BinanceClient()
    override = candles
    if override is not None and override <= 0:
        raise ValueError("candles must be greater than 0")

    table = Table("Timeframe", "Target candles", "Rows added", "Last candle")
    for tf in cfg.timeframes:
        target = override if override is not None else cfg.candles_for(tf)
        try:
            added = download_timeframe(db, client, cfg.symbol, tf, target)
            last = db.last_open_time(cfg.symbol, tf)
            table.add_row(tf, str(target), str(added), str(last or "-"))
        except Exception as exc:
            table.add_row(tf, str(target), f"ERROR: {exc}", "-")
    console.print(table)
    db.close()


def cmd_export_data() -> None:
    """Export the operational SQLite candle data to Git-friendly snapshots."""
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    table = Table("Timeframe", "Rows", "File", "Size")

    try:
        for tf in cfg.timeframes:
            df = db.load_candles(cfg.symbol, tf)
            if df.empty:
                table.add_row(tf, "0", "-", "-")
                continue
            df = df[HISTORICAL_COLUMNS].copy()
            path = HISTORICAL_DIR / f"{cfg.symbol}_{tf}.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                df.to_csv(fh, index=False)
            table.add_row(tf, str(len(df)), str(path), f"{path.stat().st_size / 1024 / 1024:.2f} MB")
    finally:
        db.close()
    console.print(table)


def cmd_restore_data() -> None:
    """Restore GitHub historical snapshots into the local SQLite database."""
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    table = Table("Timeframe", "Rows restored", "Last candle")

    try:
        for tf in cfg.timeframes:
            path = HISTORICAL_DIR / f"{cfg.symbol}_{tf}.csv.gz"
            if not path.exists():
                table.add_row(tf, "FILE NOT FOUND", "-")
                continue
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                df = pd.read_csv(fh)
            if df.empty:
                table.add_row(tf, "0", "-")
                continue
            rows = [tuple(row) for row in df[HISTORICAL_COLUMNS].itertuples(index=False, name=None)]
            restored = db.upsert_candles(rows)
            last = db.last_open_time(cfg.symbol, tf)
            table.add_row(tf, str(restored), str(last or "-"))
    finally:
        db.close()
    console.print(table)


def cmd_scan() -> None:
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    table = Table("Timeframe", "Rows", "Latest close", "Pattern vectors")
    for tf in cfg.timeframes:
        df = db.load_candles(cfg.symbol, tf, limit=max(cfg.pattern.lookbacks) + 100)
        if df.empty:
            table.add_row(tf, "0", "-", "-")
            continue
        enriched = add_features(df)
        count = 0
        for lb in cfg.pattern.lookbacks:
            if len(enriched) >= lb:
                _ = pattern_vector(enriched, len(enriched) - 1, lb)
                count += 1
        table.add_row(tf, str(len(df)), f"{df.iloc[-1]['close']:.2f}", str(count))
    console.print(table)
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ETH Market Structure Engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p_download = sub.add_parser("download")
    p_download.add_argument(
        "--candles",
        type=int,
        default=None,
        help="Override the configured candle count for every timeframe",
    )
    sub.add_parser("update")
    sub.add_parser("scan")
    sub.add_parser("export-data", help="Export SQLite candles to compressed GitHub snapshots")
    sub.add_parser("restore-data", help="Restore compressed GitHub snapshots into SQLite")
    args = parser.parse_args()
    if args.command == "init":
        cmd_init()
    elif args.command == "download":
        cmd_download(args.candles)
    elif args.command == "update":
        cmd_download(None)
    elif args.command == "scan":
        cmd_scan()
    elif args.command == "export-data":
        cmd_export_data()
    elif args.command == "restore-data":
        cmd_restore_data()


if __name__ == "__main__":
    main()
