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
from analysis.outcomes import OutcomeConfig, evaluate_matches, summarize_outcomes
from analysis.walk_forward import (
    deoverlap_outcomes,
    evaluate_endpoint_baseline,
    evaluate_walk_forward,
    summarize_outcomes_with_ci,
    summarize_similarity_buckets,
)

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
    cfg = load_config()
    if timeframe not in cfg.timeframes:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if threshold <= 0 or pivots <= 0 or top_k <= 0:
        raise ValueError("threshold, pivots and top_k must be greater than 0")
    db = MarketDatabase(cfg.database)
    try:
        candles = db.load_candles(cfg.symbol, timeframe)
        if candles.empty:
            console.print("[yellow]No candle data available.[/yellow]")
            return
        structure = build_structure(candles, threshold)
        if len(structure) < pivots * 2:
            console.print(f"[yellow]Not enough confirmed pivots: {len(structure)} available, need at least {pivots * 2}.[/yellow]")
            return
        current = structure.tail(pivots)
        matches = find_similar_structures(structure, n_pivots=pivots, top_k=top_k, minimum_similarity=minimum_similarity)
        console.print(f"\n[bold]Latest {timeframe} structure ({threshold:.2f}% ZigZag)[/bold]")
        console.print(" → ".join(current["pivot_type"].astype(str)))
        console.print(
            f"Endpoint pivot: {int(current.iloc[-1]['index'])} | "
            f"Confirmed at: {int(current.iloc[-1]['confirmation_index'])} | "
            f"Price: {float(current.iloc[-1]['price']):.2f} | Confirmed pivots: {len(structure)}"
        )
        if matches.empty:
            console.print(f"No historical matches met similarity >= {minimum_similarity:.2f}.")
            return
        table = Table("Rank", "Similarity", "End pivot", "Confirmed", "Pivot price", "Structure")
        for rank, match in matches.iterrows():
            pivot = structure.iloc[int(match["candidate_end_position"])]
            table.add_row(
                str(rank + 1), f"{float(match['similarity']):.4f}",
                str(int(match["candidate_end_index"])), str(int(match["candidate_confirmation_index"])),
                f"{float(pivot['price']):.2f}", str(match["pivot_type_sequence"]),
            )
        console.print(table)
    finally:
        db.close()


def cmd_structure_outcomes(timeframe: str, threshold: float, pivots: int, top_k: int, minimum_similarity: float) -> None:
    """Find historical matches and measure their strictly-forward outcomes."""
    cfg = load_config()
    if timeframe not in cfg.timeframes:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    db = MarketDatabase(cfg.database)
    try:
        candles = db.load_candles(cfg.symbol, timeframe)
        structure = build_structure(candles, threshold)
        if len(structure) < pivots * 2:
            console.print(f"[yellow]Not enough confirmed pivots: {len(structure)}.[/yellow]")
            return
        matches = find_similar_structures(structure, n_pivots=pivots, top_k=top_k, minimum_similarity=minimum_similarity)
        if matches.empty:
            console.print("No historical matches found.")
            return
        outcome_cfg = OutcomeConfig(
            horizons=tuple(cfg.outcomes.horizons), target_pct=cfg.outcomes.target_pct, stop_pct=cfg.outcomes.stop_pct,
        )
        outcomes = evaluate_matches(matches, candles, structure, outcome_cfg)
        summary = summarize_outcomes(outcomes, outcome_cfg.horizons)
        console.print(f"\n[bold]Historical outcomes — {timeframe}, {threshold:.2f}% ZigZag, {pivots} pivots[/bold]")
        console.print(f"Matches: {len(outcomes)} | Entry: confirmation close | Target: +{outcome_cfg.target_pct:.2f}% | Stop: -{outcome_cfg.stop_pct:.2f}%")
        detail = Table("Rank", "Similarity", "Pivot", "Entry", *[f"R{h}" for h in outcome_cfg.horizons], *[f"MFE{h}" for h in outcome_cfg.horizons], *[f"MAE{h}" for h in outcome_cfg.horizons])
        for rank, row in outcomes.iterrows():
            cells = [str(rank + 1), f"{float(row['similarity']):.4f}", str(int(row['candidate_end_index'])), str(int(row['entry_index']))]
            cells += ["-" if pd.isna(row[f"return_{h}"]) else f"{float(row[f'return_{h}']):+.2f}%" for h in outcome_cfg.horizons]
            cells += ["-" if pd.isna(row[f"mfe_{h}"]) else f"{float(row[f'mfe_{h}']):+.2f}%" for h in outcome_cfg.horizons]
            cells += ["-" if pd.isna(row[f"mae_{h}"]) else f"{float(row[f'mae_{h}']):+.2f}%" for h in outcome_cfg.horizons]
            detail.add_row(*cells)
        console.print(detail)
        summary_table = Table("Horizon", "Samples", "Target", "Stop", "Neither", "Ambiguous", "Target % (all)", "Target % (decisive)")
        for _, row in summary.iterrows():
            summary_table.add_row(
                str(int(row["horizon"])), str(int(row["samples"])), str(int(row["target"])), str(int(row["stop"])),
                str(int(row["neither"])), str(int(row["ambiguous"])),
                "-" if pd.isna(row["target_rate_all"]) else f"{row['target_rate_all']:.1%}",
                "-" if pd.isna(row["target_rate_decisive"]) else f"{row['target_rate_decisive']:.1%}",
            )
        console.print("\n[bold]Target / stop summary[/bold]")
        console.print(summary_table)
    finally:
        db.close()


def _print_summary(title: str, summary: pd.DataFrame) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    table = Table("Horizon", "Samples", "Target", "Stop", "Neither", "Ambiguous", "Target %", "Wilson 95%", "Decisive %", "Wilson 95% (decisive)")
    for _, row in summary.iterrows():
        table.add_row(
            str(int(row["horizon"])), str(int(row["samples"])), str(int(row["target"])), str(int(row["stop"])),
            str(int(row["neither"])), str(int(row["ambiguous"])),
            f"{row['target_rate_all']:.1%}",
            f"{row['target_rate_all_lower']:.1%}–{row['target_rate_all_upper']:.1%}",
            f"{row['target_rate_decisive']:.1%}",
            f"{row['target_rate_decisive_lower']:.1%}–{row['target_rate_decisive_upper']:.1%}",
        )
    console.print(table)


def cmd_walk_forward(timeframe: str, threshold: float, pivots: int, top_k: int, minimum_similarity: float, max_samples: int, spacing_candles: int | None) -> None:
    """Run causal matching plus baseline, de-overlap and similarity-bucket controls."""
    cfg = load_config()
    if timeframe not in cfg.timeframes:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if threshold <= 0 or pivots <= 0 or top_k <= 0 or max_samples <= 0:
        raise ValueError("threshold, pivots, top_k and max_samples must be greater than 0")
    db = MarketDatabase(cfg.database)
    try:
        candles = db.load_candles(cfg.symbol, timeframe)
        if candles.empty:
            console.print("[yellow]No candle data available.[/yellow]")
            return
        structure = build_structure(candles, threshold)
        outcome_cfg = OutcomeConfig(
            horizons=tuple(cfg.outcomes.horizons), target_pct=cfg.outcomes.target_pct, stop_pct=cfg.outcomes.stop_pct,
        )
        outcomes, summary = evaluate_walk_forward(
            structure, candles, n_pivots=pivots, top_k=top_k,
            minimum_similarity=minimum_similarity, outcome_config=outcome_cfg,
            max_samples=max_samples, spacing_candles=spacing_candles,
        )
        if outcomes.empty:
            console.print("No evaluable historical matches found.")
            return

        endpoint_positions = sorted({int(x) for x in outcomes["evaluation_endpoint_position"].dropna()})
        baseline = evaluate_endpoint_baseline(structure, candles, endpoint_positions, outcome_cfg)
        max_horizon = max(outcome_cfg.horizons)
        deoverlapped = deoverlap_outcomes(outcomes, max_horizon)
        de_summary = summarize_outcomes_with_ci(deoverlapped, outcome_cfg.horizons)
        bucket_summary = summarize_similarity_buckets(outcomes, outcome_cfg.horizons)
        bucket_de_summary = summarize_similarity_buckets(deoverlapped, outcome_cfg.horizons)
        baseline_summary = summarize_outcomes_with_ci(baseline, outcome_cfg.horizons)

        endpoint_count = len(endpoint_positions)
        console.print(f"\n[bold]Walk-forward evaluation — {timeframe}, {threshold:.2f}% ZigZag, {pivots} pivots[/bold]")
        console.print(
            f"Evaluation endpoints: {endpoint_count} | Match outcomes: {len(outcomes)} | "
            f"De-overlapped matches: {len(deoverlapped)} | Top-K: {top_k} | "
            f"Min similarity: {minimum_similarity:.2f} | Entry: match confirmation close"
        )
        _print_summary("Matched outcomes — raw", summary)
        _print_summary("Baseline — actual evaluation endpoints (no matching)", baseline_summary)
        _print_summary("Matched outcomes — de-overlapped", de_summary)

        if not bucket_summary.empty:
            console.print("\n[bold]Similarity buckets — raw matches[/bold]")
            table = Table("Similarity", "Horizon", "Samples", "Target %", "Wilson 95%", "Decisive %", "Wilson 95% (decisive)")
            for _, row in bucket_summary.iterrows():
                table.add_row(
                    str(row["similarity_bucket"]), str(int(row["horizon"])), str(int(row["samples"])),
                    f"{row['target_rate_all']:.1%}",
                    f"{row['target_rate_all_lower']:.1%}–{row['target_rate_all_upper']:.1%}",
                    f"{row['target_rate_decisive']:.1%}",
                    f"{row['target_rate_decisive_lower']:.1%}–{row['target_rate_decisive_upper']:.1%}",
                )
            console.print(table)
        if not bucket_de_summary.empty:
            console.print("\n[bold]Similarity buckets — de-overlapped matches[/bold]")
            table = Table("Similarity", "Horizon", "Samples", "Target %", "Wilson 95%", "Decisive %", "Wilson 95% (decisive)")
            for _, row in bucket_de_summary.iterrows():
                table.add_row(
                    str(row["similarity_bucket"]), str(int(row["horizon"])), str(int(row["samples"])),
                    f"{row['target_rate_all']:.1%}",
                    f"{row['target_rate_all_lower']:.1%}–{row['target_rate_all_upper']:.1%}",
                    f"{row['target_rate_decisive']:.1%}",
                    f"{row['target_rate_decisive_lower']:.1%}–{row['target_rate_decisive_upper']:.1%}",
                )
            console.print(table)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ETH Market Structure Engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p_download = sub.add_parser("download")
    p_download.add_argument("--candles", type=int, default=None)
    sub.add_parser("update")
    sub.add_parser("scan")
    p_match = sub.add_parser("structure-match")
    p_match.add_argument("--timeframe", default="1h")
    p_match.add_argument("--threshold", type=float, default=1.0)
    p_match.add_argument("--pivots", type=int, default=8)
    p_match.add_argument("--top-k", type=int, default=10)
    p_match.add_argument("--minimum-similarity", type=float, default=None)
    p_outcome = sub.add_parser("structure-outcomes", help="Match structures and measure strictly-forward outcomes")
    p_outcome.add_argument("--timeframe", default="1h")
    p_outcome.add_argument("--threshold", type=float, default=1.0)
    p_outcome.add_argument("--pivots", type=int, default=8)
    p_outcome.add_argument("--top-k", type=int, default=10)
    p_outcome.add_argument("--minimum-similarity", type=float, default=None)
    p_walk = sub.add_parser("walk-forward", help="Evaluate endpoints with baseline, overlap and similarity controls")
    p_walk.add_argument("--timeframe", default="1h")
    p_walk.add_argument("--threshold", type=float, default=1.0)
    p_walk.add_argument("--pivots", type=int, default=8)
    p_walk.add_argument("--top-k", type=int, default=10)
    p_walk.add_argument("--minimum-similarity", type=float, default=None)
    p_walk.add_argument("--max-samples", type=int, default=250)
    p_walk.add_argument("--spacing-candles", type=int, default=None)
    sub.add_parser("export-data")
    sub.add_parser("restore-data")
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "init": cmd_init()
    elif args.command == "download": cmd_download(args.candles)
    elif args.command == "update": cmd_download(None)
    elif args.command == "scan": cmd_scan()
    elif args.command == "structure-match": cmd_structure_match(args.timeframe, args.threshold, args.pivots, args.top_k, cfg.pattern.minimum_similarity if args.minimum_similarity is None else args.minimum_similarity)
    elif args.command == "structure-outcomes": cmd_structure_outcomes(args.timeframe, args.threshold, args.pivots, args.top_k, cfg.pattern.minimum_similarity if args.minimum_similarity is None else args.minimum_similarity)
    elif args.command == "walk-forward": cmd_walk_forward(args.timeframe, args.threshold, args.pivots, args.top_k, cfg.pattern.minimum_similarity if args.minimum_similarity is None else args.minimum_similarity, args.max_samples, args.spacing_candles)
    elif args.command == "export-data": cmd_export_data()
    elif args.command == "restore-data": cmd_restore_data()


if __name__ == "__main__":
    main()
