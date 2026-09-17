"""Continuous candle-behaviour fingerprint research.

Builds a descriptive, multi-component fingerprint for the latest candle window
and compares it with historical windows using the existing candle similarity
engine. This module intentionally performs no trading, signal, target, stop,
outcome, or future-direction prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import candle_features, candle_sequence_similarity
from config import load_config
from data.database import MarketDatabase

RANGE_SCALE = 5.0


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        raw = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if raw.empty:
        raise ValueError(f"No data available for {timeframe}")
    return raw.reset_index(drop=True)


def _rr(f):
    return f["range_ratio"] * RANGE_SCALE


def _shift(series):
    half = max(1, len(series) // 2)
    return float(series.iloc[-half:].mean() - series.iloc[:half].mean())


def _fraction(mask):
    return float(mask.mean()) if len(mask) else 0.0


def fingerprint(f: pd.DataFrame) -> dict:
    """Return continuous descriptive components plus compact human-readable codes."""
    if len(f) < 3:
        raise ValueError("At least 3 candles are required")

    rr = _rr(f)
    direction = f.direction.to_numpy(dtype=float)
    body = f.body_pct
    close_loc = f.close_location
    alt = _fraction((direction[1:] * direction[:-1]) < 0)
    bull = _fraction(f.direction > 0)
    bear = _fraction(f.direction < 0)
    upper = _fraction(f.upper_wick_pct >= 0.35)
    lower = _fraction(f.lower_wick_pct >= 0.35)
    compression = _fraction(rr < 0.85)
    expansion = _fraction(rr > 1.15)
    small_body = _fraction(body <= 0.25)
    large_body = _fraction(body >= 0.60)
    body_shift = _shift(body)
    range_shift = _shift(rr)
    close_shift = _shift(close_loc)
    direction_shift = _shift(pd.Series((direction > 0).astype(float)))

    third = max(1, len(f) // 3)
    first = f.iloc[:third]
    middle = f.iloc[third:2 * third]
    last = f.iloc[-third:]
    first_rr, last_rr = _rr(first).mean(), _rr(last).mean()
    expansion_transition = float(last_rr - first_rr)
    first_alt = _fraction((first.direction.to_numpy()[1:] * first.direction.to_numpy()[:-1]) < 0) if len(first) > 1 else 0.0
    last_alt = _fraction((last.direction.to_numpy()[1:] * last.direction.to_numpy()[:-1]) < 0) if len(last) > 1 else 0.0

    balance = 1.0 - abs(bull - 0.5) * 2.0
    rejection_balance = 1.0 - abs(upper - lower)
    wick_bias = lower - upper

    def level(x, lo, hi):
        if x <= lo:
            return "LOW"
        if x >= hi:
            return "HIGH"
        return "MID"

    balance_code = "BALANCED" if balance >= 0.70 else ("BULL_BIASED" if bull > 0.5 else "BEAR_BIASED")
    range_code = "EXPANDING" if range_shift >= 0.20 else ("COMPRESSING" if range_shift <= -0.20 else "STABLE")
    wick_code = "LOWER_WICK" if wick_bias >= 0.12 else ("UPPER_WICK" if wick_bias <= -0.12 else "WICK_BALANCED")
    direction_code = "BULL_SHIFT" if direction_shift >= 0.20 else ("BEAR_SHIFT" if direction_shift <= -0.20 else "DIRECTION_BALANCED")
    alternation_code = "HIGH_ALT" if alt >= 0.55 else ("LOW_ALT" if alt <= 0.30 else "MID_ALT")
    body_code = "LARGE_BODY" if body.mean() >= 0.60 else ("SMALL_BODY" if body.mean() <= 0.25 else "MID_BODY")
    transition_code = "COMP_TO_EXPAND" if expansion_transition >= 0.35 else ("EXP_TO_COMPRESS" if expansion_transition <= -0.35 else "RANGE_STABLE")
    close_code = "CLOSE_HIGH" if close_loc.mean() >= 0.65 else ("CLOSE_LOW" if close_loc.mean() <= 0.35 else "CLOSE_MID")

    components = {
        "direction_balance": bull - bear,
        "alternation": alt,
        "body_level": float(body.mean()),
        "body_shift": body_shift,
        "small_body_fraction": small_body,
        "large_body_fraction": large_body,
        "upper_rejection": upper,
        "lower_rejection": lower,
        "rejection_balance": rejection_balance,
        "wick_bias": wick_bias,
        "range_level": float(rr.mean()),
        "range_shift": range_shift,
        "compression_fraction": compression,
        "expansion_fraction": expansion,
        "expansion_transition": expansion_transition,
        "close_location": float(close_loc.mean()),
        "close_shift": close_shift,
        "direction_shift": direction_shift,
        "first_third_alternation": first_alt,
        "last_third_alternation": last_alt,
        "bullish_fraction": bull,
        "bearish_fraction": bear,
        "middle_third_range": float(_rr(middle).mean()) if len(middle) else np.nan,
    }
    vector = np.array([components[k] for k in sorted(components)], dtype=float)
    digest = hashlib.sha1(np.round(vector, 2).tobytes()).hexdigest()[:10].upper()
    pattern_code = " | ".join([balance_code, range_code, wick_code, direction_code, alternation_code, body_code, transition_code, close_code])
    return {**components, "pattern_code": pattern_code, "fingerprint_id": f"FP-{digest}"}


def rank_windows(features, current_end, window, top_k):
    current = features.iloc[current_end - window + 1:current_end + 1]
    rows = []
    for end in range(window - 1, current_end - window + 1):
        candidate = features.iloc[end - window + 1:end + 1]
        score = candle_sequence_similarity(current, candidate)
        rows.append({"candidate_start": end - window + 1, "candidate_end": end, "similarity": float(score)})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def context_summary(features, start, end, context):
    before = features.iloc[max(0, start - context):start]
    after = features.iloc[end + 1:min(len(features), end + 1 + context)]
    return {
        "before_bullish": _fraction(before.direction > 0),
        "matched_bullish": _fraction(features.iloc[start:end + 1].direction > 0),
        "after_bullish": _fraction(after.direction > 0) if len(after) else np.nan,
        "before_range": float(_rr(before).mean()) if len(before) else np.nan,
        "matched_range": float(_rr(features.iloc[start:end + 1]).mean()),
        "after_range": float(_rr(after).mean()) if len(after) else np.nan,
        "before_alternation": _fraction((before.direction.to_numpy()[1:] * before.direction.to_numpy()[:-1]) < 0) if len(before) > 1 else np.nan,
        "matched_alternation": _fraction((features.iloc[start:end + 1].direction.to_numpy()[1:] * features.iloc[start:end + 1].direction.to_numpy()[:-1]) < 0),
        "after_alternation": _fraction((after.direction.to_numpy()[1:] * after.direction.to_numpy()[:-1]) < 0) if len(after) > 1 else np.nan,
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

    current_rows, match_rows, context_rows = [], [], []
    for tf in timeframes:
        raw = load_market(cfg, tf)
        if len(raw) < args.window * 2 + args.context:
            continue
        f = candle_features(raw)
        current_end = len(f) - 1
        start = current_end - args.window + 1
        current = f.iloc[start:current_end + 1]
        current_fp = fingerprint(current)
        current_rows.append({"timeframe": tf, **current_fp})

        ranked = rank_windows(f, current_end, args.window, args.top_k)
        for rank, row in ranked.iterrows():
            s, e = int(row.candidate_start), int(row.candidate_end)
            match = f.iloc[s:e + 1]
            fp = fingerprint(match)
            match_rows.append({"timeframe": tf, "rank": rank + 1, "candidate_start": s, "candidate_end": e, "similarity": float(row.similarity), **fp})
            context_rows.append({"timeframe": tf, "rank": rank + 1, "candidate_start": s, "candidate_end": e, "similarity": float(row.similarity), "current_fingerprint_id": current_fp["fingerprint_id"], "matched_fingerprint_id": fp["fingerprint_id"], "matched_pattern_code": fp["pattern_code"], **context_summary(f, s, e, args.context)})

    current_df = pd.DataFrame(current_rows)
    matches_df = pd.DataFrame(match_rows)
    context_df = pd.DataFrame(context_rows)
    current_df.to_csv(charts / "candle_fingerprint_current.csv", index=False)
    matches_df.to_csv(charts / "candle_fingerprint_matches.csv", index=False)
    context_df.to_csv(charts / "candle_fingerprint_context.csv", index=False)

    if not matches_df.empty:
        cluster = (matches_df.groupby(["timeframe", "pattern_code"]).agg(count=("rank", "size"), mean_similarity=("similarity", "mean"), best_similarity=("similarity", "max")).reset_index().sort_values(["timeframe", "count", "mean_similarity"], ascending=[True, False, False]))
        cluster.to_csv(charts / "candle_fingerprint_cluster_summary.csv", index=False)

        top = matches_df[matches_df["rank"] <= min(10, args.top_k)]
        fig, ax = plt.subplots(figsize=(14, 7))
        for tf, g in top.groupby("timeframe"):
            ax.plot(g["rank"], g["similarity"], marker="o", label=tf)
        ax.set_title("Candle Behaviour Fingerprint — Historical Similarity Profile")
        ax.set_xlabel("Historical neighbour rank")
        ax.set_ylabel("Candle similarity")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "candle_fingerprint_similarity.png", dpi=160)
        plt.close(fig)

        component_names = ["direction_balance", "alternation", "body_level", "body_shift", "upper_rejection", "lower_rejection", "range_level", "range_shift", "compression_fraction", "expansion_fraction", "close_location", "close_shift", "direction_shift"]
        fig, ax = plt.subplots(figsize=(15, 7))
        plot_df = current_df.set_index("timeframe")[component_names]
        plot_df.T.plot(kind="bar", ax=ax)
        ax.set_title("Current Candle Behaviour Fingerprint Components")
        ax.set_xlabel("Fingerprint component")
        ax.set_ylabel("Measured value")
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        fig.savefig(charts / "candle_fingerprint_profile.png", dpi=160)
        plt.close(fig)

    print("CANDLE BEHAVIOUR FINGERPRINT RESEARCH")
    print("Continuous descriptive fingerprint only — no signals, trades, targets, stops, outcomes, or predictions.")
    print("\nCURRENT FINGERPRINTS")
    print(current_df[["timeframe", "fingerprint_id", "pattern_code", "bullish_fraction", "bearish_fraction", "alternation", "body_level", "range_level", "range_shift", "upper_rejection", "lower_rejection", "direction_shift", "expansion_transition"]].to_string(index=False))
    print("\nTOP HISTORICAL FINGERPRINT PATTERNS")
    if not matches_df.empty:
        print(matches_df.groupby(["timeframe", "pattern_code"]).agg(count=("rank", "size"), mean_similarity=("similarity", "mean"), best_similarity=("similarity", "max")).reset_index().sort_values(["timeframe", "count", "mean_similarity"], ascending=[True, False, False]).to_string(index=False))
    print("\nSaved fingerprint outputs in charts/")


if __name__ == "__main__":
    main()
