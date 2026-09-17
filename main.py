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
from structure.representation import build_structure
from structure.similarity import find_similar_structures

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


def cmd_structure_match(timeframe: str, threshold: float, pivots: int, top_k: int, minimum_similarity: float) -> None:
    """Find historical structures similar to the latest confirmed structure."""
    cfg = load_config()
    if timeframe not in cfg.timeframes:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    if pivots <= 0 or top_k <= 0:
        raise ValueError("pivots and top_k must be greater than 0")

    db = MarketDatabase(cfg.database)
    try:
        candles = db.load_candles(cfg.symbol, timeframe)
        if candles.empty:
            console.print("[yellow]No candle data available.[/yellow]")
            return

        structure = build_structure(candles, threshold)
        if len(structure) < pivots * 2:
            console.print(
                f"[yellow]Not enough confirmed pivots: {len(structure)} available, "
                f"need at least {pivots * 2}.[/yellow]"
            )
            return

        current = structure.tail(pivots)
        matches = find_similar_structures(
            structure,
            n_pivots=pivots,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )

        console.print(f"\n[bold]Latest {timeframe} structure ({threshold:.2f}% ZigZag)[/bold]")
        console.print(" → ".join(current["pivot_type"].astype(str)))
        console.print(
            f"Endpoint pivot: {int(current.iloc[-1]['index'])} | "
            f"Price: {float(current.iloc[-1]['price']):.2f} | "
            f"Confirmed pivots: {len(structure)}"
        )

        table = Table("Rank", "Similarity", "End pivot", "Pivot price", "Structure")
        for rank, match in matches.iterrows():
            end_pos = int(match["candidate_end_position"])
            pivot = structure.iloc[end_pos]
            table.add_row(
                str(rank + 1),
                f"{float(match['similarity']):.4f}",
                str(int(match["candidate_end_index"])),
                f"{float(pivot['price']):.2f}",
                str(match["pivot_type_sequence"]),
            )
        if matches.empty:
            console.print(f"No historical matches met similarity >= {minimum_similarity:.2f}.")
        else:
            console.print(table)
    finally:
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
    p_match = sub.add_parser("structure-match", help="Find historical structures similar to the latest structure")
    p_match.add_argument("--timeframe", default="1h")
    p_match.add_argument("--threshold", type=float, default=1.0)
    p_match.add_argument("--pivots", type=int, default=8)
    p_match.add_argument("--top-k", type=int, default=10)
    p_match.add_argument("--minimum-similarity", type=float, default=None)
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
    elif args.command == "structure-match":
        cfg = load_config()
        threshold = args.threshold
        minimum = cfg.pattern.minimum_similarity if args.minimum_similarity is None else args.minimum_similarity
        cmd_structure_match(args.timeframe, threshold, args.pivots, args.top_k, minimum)
    elif args.command == "export-data":
        cmd_export_data()
    elif args.command == "restore-data":
        cmd_restore_data()


if __name__ == "__main__":
    main()
