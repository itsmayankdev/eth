"""Descriptive candle-behaviour archetype research.

Groups historical candle windows into recurring descriptive behaviour profiles.
This module intentionally performs no trading, signal, target, stop, outcome,
or future-direction prediction.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import candle_features, candle_sequence_similarity
from config import load_config
from data.database import MarketDatabase

RANGE_SCALE = 5.0
COMPRESSION_RATIO = 0.85
EXPANSION_RATIO = 1.15
SMALL_BODY = 0.25
LARGE_BODY = 0.60
REJECTION_WICK = 0.35


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        raw = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if raw.empty:
        raise ValueError(f"No data available for {timeframe}")
    return raw.reset_index(drop=True)


def _raw_range_ratio(f: pd.DataFrame) -> pd.Series:
    """Convert the similarity module's 0..1 scaled range_ratio back to 0..5."""
    return f["range_ratio"] * RANGE_SCALE


def _directional_shift(f: pd.DataFrame) -> float:
    half = max(1, len(f) // 2)
    first, second = f.iloc[:half], f.iloc[-half:]
    return float((second.direction > 0).mean() - (first.direction > 0).mean())


def _range_shift(f: pd.DataFrame) -> float:
    half = max(1, len(f) // 2)
    first, second = f.iloc[:half], f.iloc[-half:]
    return float(_raw_range_ratio(second).mean() - _raw_range_ratio(first).mean())


def classify_micro_behaviour(f: pd.DataFrame) -> str:
    """Fine-grained descriptive label for the internal candle anatomy."""
    if len(f) < 2:
        return "INSUFFICIENT_DATA"
    rr = _raw_range_ratio(f)
    bull = float((f.direction > 0).mean())
    alt = float(((f.direction.to_numpy()[1:] * f.direction.to_numpy()[:-1]) < 0).mean())
    compression = float((rr < COMPRESSION_RATIO).mean())
    expansion = float((rr > EXPANSION_RATIO).mean())
    upper = float((f.upper_wick_pct >= REJECTION_WICK).mean())
    lower = float((f.lower_wick_pct >= REJECTION_WICK).mean())
    rejection = max(upper, lower)
    shift = _directional_shift(f)
    range_shift = _range_shift(f)
    mean_body = float(f.body_pct.mean())

    if compression >= 0.55 and abs(bull - 0.5) <= 0.15 and rejection >= 0.35:
        return "WICK_REJECTION_COMPRESSION"
    if compression >= 0.55 and abs(bull - 0.5) <= 0.15:
        return "BALANCED_COMPRESSION"
    if compression >= 0.45 and abs(shift) >= 0.25:
        return "DIRECTIONAL_COMPRESSION"
    if expansion >= 0.35 and range_shift >= 0.10:
        return "RANGE_EXPANSION"
    if rejection >= 0.45 and abs(bull - 0.5) <= 0.25:
        return "REJECTION_BALANCE"
    if shift >= 0.25 and range_shift >= 0.0:
        return "BULLISH_SHIFT"
    if shift <= -0.25 and range_shift >= 0.0:
        return "BEARISH_SHIFT"
    if bull >= 0.70 and alt <= 0.35 and mean_body >= LARGE_BODY:
        return "PERSISTENT_BULLISH_BODY"
    if bull <= 0.30 and alt <= 0.35 and mean_body >= LARGE_BODY:
        return "PERSISTENT_BEARISH_BODY"
    if alt >= 0.55:
        return "HIGH_ALTERNATION"
    if mean_body <= SMALL_BODY:
        return "SMALL_BODY_MIX"
    return "MIXED_MICRO_BEHAVIOUR"


def classify_sequence(f: pd.DataFrame) -> str:
    """Assign a broader behaviour archetype using correctly scaled ranges."""
    n = len(f)
    if n < 2:
        return "INSUFFICIENT_DATA"
    bull = float((f.direction > 0).mean())
    alt = float(((f.direction.to_numpy()[1:] * f.direction.to_numpy()[:-1]) < 0).mean())
    rr = _raw_range_ratio(f)
    compression = float((rr < COMPRESSION_RATIO).mean())
    expansion = float((rr > EXPANSION_RATIO).mean())
    shift = _directional_shift(f)
    range_shift = _range_shift(f)
    rejection = float(((f.lower_wick_pct >= REJECTION_WICK) | (f.upper_wick_pct >= REJECTION_WICK)).mean())

    if shift >= 0.30 and range_shift >= 0.10:
        return "BEARISH_TO_BULLISH_SHIFT"
    if shift <= -0.30 and range_shift >= 0.10:
        return "BULLISH_TO_BEARISH_SHIFT"
    if compression >= 0.55 and abs(bull - 0.50) <= 0.15:
        return "SIDEWAYS_COMPRESSION"
    if abs(bull - 0.50) <= 0.20 and max(bull, 1 - bull) >= 0.65 and range_shift >= 0.10:
        return "SIDEWAYS_TO_DIRECTIONAL"
    if expansion >= 0.35 and range_shift >= 0.10:
        return "EXPANSION_SEQUENCE"
    if compression >= 0.55 and range_shift <= -0.10:
        return "COMPRESSION_SEQUENCE"
    if rejection >= 0.45 and abs(bull - 0.50) <= 0.25:
        return "REJECTION_MIX"
    if bull >= 0.70 and alt <= 0.35:
        return "PERSISTENT_BULLISH_BODY"
    if bull <= 0.30 and alt <= 0.35:
        return "PERSISTENT_BEARISH_BODY"
    if alt >= 0.55:
        return "HIGH_ALTERNATION"
    return "MIXED_BEHAVIOUR"


def summarize(f: pd.DataFrame) -> dict:
    half = max(1, len(f) // 2)
    first, second = f.iloc[:half], f.iloc[-half:]
    rr = _raw_range_ratio(f)
    return {
        "candles": len(f),
        "bullish_fraction": float((f.direction > 0).mean()),
        "bearish_fraction": float((f.direction < 0).mean()),
        "alternation_fraction": float(((f.direction.to_numpy()[1:] * f.direction.to_numpy()[:-1]) < 0).mean()) if len(f) > 1 else 0.0,
        "compression_fraction": float((rr < COMPRESSION_RATIO).mean()),
        "expansion_fraction": float((rr > EXPANSION_RATIO).mean()),
        "upper_rejection_fraction": float((f.upper_wick_pct >= REJECTION_WICK).mean()),
        "lower_rejection_fraction": float((f.lower_wick_pct >= REJECTION_WICK).mean()),
        "mean_body_pct": float(f.body_pct.mean()),
        "mean_range_ratio": float(rr.mean()),
        "first_half_bullish": float((first.direction > 0).mean()),
        "second_half_bullish": float((second.direction > 0).mean()),
        "directional_shift": float((second.direction > 0).mean() - (first.direction > 0).mean()),
        "range_shift": float(_raw_range_ratio(second).mean() - _raw_range_ratio(first).mean()),
        "body_shift": float(second.body_pct.mean() - first.body_pct.mean()),
        "archetype": classify_sequence(f),
        "micro_behaviour": classify_micro_behaviour(f),
    }


def rank_windows(features, current_end, window, top_k):
    current = features.iloc[current_end - window + 1:current_end + 1]
    rows = []
    for end in range(window - 1, current_end - window + 1):
        candidate = features.iloc[end - window + 1:end + 1]
        score = candle_sequence_similarity(current, candidate)
        rows.append({"start": end - window + 1, "end": end, "similarity": float(score)})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def transition(before, match, after):
    return {
        "before_archetype": classify_sequence(before) if len(before) else "NONE",
        "matched_archetype": classify_sequence(match),
        "after_archetype": classify_sequence(after) if len(after) else "NONE",
        "before_micro_behaviour": classify_micro_behaviour(before) if len(before) else "NONE",
        "matched_micro_behaviour": classify_micro_behaviour(match),
        "after_micro_behaviour": classify_micro_behaviour(after) if len(after) else "NONE",
        "before_bullish": float((before.direction > 0).mean()) if len(before) else np.nan,
        "matched_bullish": float((match.direction > 0).mean()),
        "after_bullish": float((after.direction > 0).mean()) if len(after) else np.nan,
        "before_range": float(_raw_range_ratio(before).mean()) if len(before) else np.nan,
        "matched_range": float(_raw_range_ratio(match).mean()),
        "after_range": float(_raw_range_ratio(after).mean()) if len(after) else np.nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--context", type=int, default=10)
    p.add_argument("--all-timeframes", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    timeframes = cfg.timeframes if args.all_timeframes else [args.timeframe]
    charts = Path("charts")
    charts.mkdir(exist_ok=True)
    for item in charts.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        else:
            shutil.rmtree(item)

    all_summary, all_context, current_rows = [], [], []
    for tf in timeframes:
        raw = load_market(cfg, tf)
        if len(raw) < args.window * 2 + args.context:
            continue
        f = candle_features(raw)
        current_end = len(f) - 1
        current_start = current_end - args.window + 1
        current = f.iloc[current_start:current_end + 1]
        current_summary = summarize(current)
        current_rows.append({"timeframe": tf, **current_summary})
        ranked = rank_windows(f, current_end, args.window, args.top_k)
        for rank, row in ranked.iterrows():
            start, end = int(row.start), int(row.end)
            match = f.iloc[start:end + 1]
            summary = summarize(match)
            all_summary.append({"timeframe": tf, "rank": rank + 1, "candidate_start": start, "candidate_end": end, "similarity": float(row.similarity), **summary})
            before = f.iloc[max(0, start - args.context):start]
            after = f.iloc[end + 1:min(len(f), end + 1 + args.context)]
            all_context.append({"timeframe": tf, "rank": rank + 1, "similarity": float(row.similarity), "candidate_start": start, "candidate_end": end, **transition(before, match, after)})

    summary_df = pd.DataFrame(all_summary)
    context_df = pd.DataFrame(all_context)
    current_df = pd.DataFrame(current_rows)
    summary_df.to_csv(charts / "candle_archetype_matches.csv", index=False)
    context_df.to_csv(charts / "candle_archetype_context.csv", index=False)
    current_df.to_csv(charts / "candle_archetype_current.csv", index=False)

    if not summary_df.empty:
        counts = summary_df.groupby(["timeframe", "archetype"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(14, 7))
        pivot = counts.pivot(index="archetype", columns="timeframe", values="count").fillna(0)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Historical Candle Behaviour Archetypes — Top Neighbours")
        ax.set_xlabel("Descriptive archetype")
        ax.set_ylabel("Count among selected historical neighbours")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        fig.savefig(charts / "candle_archetype_distribution.png", dpi=160)
        plt.close(fig)

        top = summary_df[summary_df["rank"] <= min(10, args.top_k)].copy()
        fig, ax = plt.subplots(figsize=(13, 6))
        for tf, g in top.groupby("timeframe"):
            ax.plot(g["rank"], g["similarity"], marker="o", label=tf)
        ax.set_title("Similarity Profile Across Top Historical Neighbours")
        ax.set_xlabel("Historical neighbour rank")
        ax.set_ylabel("Candle similarity")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "candle_archetype_similarity_profile.png", dpi=160)
        plt.close(fig)

    print("CANDLE BEHAVIOUR ARCHETYPE RESEARCH")
    print("Descriptive only — no signals, targets, stops, outcomes, or predictions.")
    print("Range ratios are restored to their original 0..5 scale for behaviour classification.")
    print("\nCURRENT ARCHETYPE")
    print(current_df[["timeframe", "archetype", "micro_behaviour", "bullish_fraction", "bearish_fraction", "compression_fraction", "expansion_fraction", "directional_shift", "range_shift"]].to_string(index=False))
    print("\nHISTORICAL ARCHETYPE COUNTS")
    if not summary_df.empty:
        print(summary_df.groupby(["timeframe", "archetype"]).size().reset_index(name="count").to_string(index=False))
    print("\nHISTORICAL MICRO-BEHAVIOUR COUNTS")
    if not summary_df.empty:
        print(summary_df.groupby(["timeframe", "micro_behaviour"]).size().reset_index(name="count").to_string(index=False))
    print("\nSaved fresh archetype research outputs in charts/")


if __name__ == "__main__":
    main()
