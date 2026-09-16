from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import load_config
from data.binance import BinanceClient
from data.database import MarketDatabase
from data.downloader import download_timeframe
from features.feature_engine import add_features, pattern_vector

console = Console()


def cmd_init() -> None:
    cfg = load_config()
    MarketDatabase(cfg.database).close()
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    console.print(f"[green]Database ready:[/green] {cfg.database}")


def cmd_download(candles: int | None) -> None:
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    client = BinanceClient()
    target = candles if candles is not None else cfg.historical_candles
    if target <= 0:
        raise ValueError("candles must be greater than 0")

    table = Table("Timeframe", "Rows added", "Last candle")
    for tf in cfg.timeframes:
        try:
            added = download_timeframe(db, client, cfg.symbol, tf, target)
            last = db.last_open_time(cfg.symbol, tf)
            table.add_row(tf, str(added), str(last or "-"))
        except Exception as exc:
            table.add_row(tf, f"ERROR: {exc}", "-")
    console.print(table)
    db.close()


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
        help="Historical closed candles to load on an empty timeframe (default: config value)",
    )
    sub.add_parser("update")
    sub.add_parser("scan")
    args = parser.parse_args()
    if args.command == "init":
        cmd_init()
    elif args.command == "download":
        cmd_download(args.candles)
    elif args.command == "update":
        # Empty timeframes receive the configured initial history; existing
        # timeframes receive only candles after their latest stored candle.
        cmd_download(None)
    elif args.command == "scan":
        cmd_scan()


if __name__ == "__main__":
    main()
