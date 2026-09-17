"""Multi-timeframe descriptive candle-behaviour research.

Runs the same candle-state representation across every configured timeframe and
compares the current sequence with historical candle-shape neighbours. This is
research-only: no trades, targets, stops, signals, or outcome predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_behavior_states import classify_candles, summarize
from candle_similarity import candle_features, rank_candle_sequences
from config import load_config
from data.database import MarketDatabase


def analyse_timeframe(db: MarketDatabase, symbol: str, timeframe: str, window: int, top_k: int) -> tuple[dict, pd.DataFrame]:
    raw = db.load_candles(symbol, timeframe).reset_index(drop=True)
    if len(raw) < window + 5:
        raise ValueError(f"{timeframe}: not enough candles ({len(raw)})")

    features = candle_features(raw)
    states = classify_candles(raw, features)
    current_start = len(raw) - window
    current = states.iloc[current_start:].copy()

    ranked = rank_candle_sequences(
        features,
        current_start=current_start,
        current_end=len(raw) - 1,
        candidate_ends=range(window - 1, current_start),
    ).head(top_k)

    s = summarize(current)
    best_similarity = float(ranked.iloc[0]["similarity"]) if len(ranked) else np.nan
    row = {
        "timeframe": timeframe,
        "candles_available": len(raw),
        "window": window,
        "best_similarity": best_similarity,
        **s,
    }
    return row, ranked


def plot_profiles(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("bullish_fraction", "Bullish fraction"),
        ("bearish_fraction", "Bearish fraction"),
        ("compression_fraction", "Compression fraction"),
        ("expansion_fraction", "Expansion fraction"),
        ("upper_rejection_fraction", "Upper rejection fraction"),
        ("lower_rejection_fraction", "Lower rejection fraction"),
        ("directional_shift", "Directional shift"),
        ("range_shift", "Range shift"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.ravel()
    x = np.arange(len(summary))
    labels = summary["timeframe"].tolist()
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(x, summary[column].to_numpy(dtype=float))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Current Candle Behaviour Across Timeframes", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-timeframe candle behaviour research")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeframes", nargs="*", default=None)
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    timeframes = args.timeframes or list(cfg.timeframes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = MarketDatabase(cfg.database)
    rows = []
    best_rows = []
    try:
        for timeframe in timeframes:
            try:
                row, ranked = analyse_timeframe(db, cfg.symbol, timeframe, args.window, args.top_k)
                rows.append(row)
                for rank, item in enumerate(ranked.to_dict("records"), start=1):
                    best_rows.append({
                        "timeframe": timeframe,
                        "rank": rank,
                        **item,
                    })
                print(
                    f"{timeframe}: profile={row['transition_profile']} "
                    f"bull={row['bullish_fraction']:.2f} "
                    f"bear={row['bearish_fraction']:.2f} "
                    f"compression={row['compression_fraction']:.2f} "
                    f"expansion={row['expansion_fraction']:.2f} "
                    f"best_similarity={row['best_similarity']:.4f}"
                )
            except Exception as exc:
                print(f"{timeframe}: ERROR: {exc}")
    finally:
        db.close()

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No timeframe completed successfully")

    summary.to_csv(out_dir / "candle_behavior_multitimeframe_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "candle_behavior_multitimeframe_matches.csv", index=False)
    plot_profiles(summary, out_dir / "candle_behavior_multitimeframe.png")

    print("\nMULTI-TIMEFRAME SUMMARY")
    cols = [
        "timeframe", "transition_profile", "bullish_fraction", "bearish_fraction",
        "compression_fraction", "expansion_fraction", "directional_shift",
        "range_shift", "best_similarity",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nSaved outputs in {out_dir}/")


if __name__ == "__main__":
    main()
