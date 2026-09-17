"""Descriptive candle-behaviour state research.

This module intentionally does not create trades, targets, stops, signals, or
outcome predictions. It describes OHLC candle anatomy and sequence transitions
so historical windows can be compared as behaviour patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import candle_features, rank_candle_sequences
from config import load_config
from storage import MarketDatabase


RANGE_COMPRESSION = 0.75
RANGE_EXPANSION = 1.25
SMALL_BODY = 0.25
LARGE_BODY = 0.60
REJECTION_WICK = 0.35


def classify_candles(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return multi-label, human-readable states for every candle."""
    out = raw[["open", "high", "low", "close"]].copy()
    body = features["body_pct"].to_numpy()
    upper = features["upper_wick_pct"].to_numpy()
    lower = features["lower_wick_pct"].to_numpy()
    close_loc = features["close_location"].to_numpy()
    # candle_similarity stores range/median divided by five; restore the
    # intuitive range/median ratio here for state thresholds.
    range_ratio = features["range_ratio"].to_numpy() * 5.0
    direction = features["direction"].to_numpy()

    states = []
    for i in range(len(out)):
        tags = []
        if direction[i] > 0:
            tags.append("BULLISH")
        elif direction[i] < 0:
            tags.append("BEARISH")
        else:
            tags.append("DOJI")

        if body[i] < SMALL_BODY:
            tags.append("SMALL_BODY")
        elif body[i] >= LARGE_BODY:
            tags.append("LARGE_BODY")
        else:
            tags.append("MEDIUM_BODY")

        if range_ratio[i] < RANGE_COMPRESSION:
            tags.append("COMPRESSION")
        elif range_ratio[i] > RANGE_EXPANSION:
            tags.append("EXPANSION")
        else:
            tags.append("NORMAL_RANGE")

        if upper[i] >= REJECTION_WICK:
            tags.append("UPPER_REJECTION")
        if lower[i] >= REJECTION_WICK:
            tags.append("LOWER_REJECTION")

        if close_loc[i] >= 0.75:
            tags.append("CLOSE_HIGH")
        elif close_loc[i] <= 0.25:
            tags.append("CLOSE_LOW")

        if direction[i] > 0 and body[i] >= LARGE_BODY and close_loc[i] >= 0.70:
            tags.append("STRONG_BULL")
        if direction[i] < 0 and body[i] >= LARGE_BODY and close_loc[i] <= 0.30:
            tags.append("STRONG_BEAR")

        states.append("|".join(tags))

    out["state"] = states
    out["body_pct"] = features["body_pct"].to_numpy()
    out["range_ratio"] = range_ratio
    out["upper_wick_pct"] = upper
    out["lower_wick_pct"] = lower
    out["close_location"] = close_loc
    out["direction"] = direction
    return out


def _summary(window: pd.DataFrame) -> dict:
    """Summarise behaviour without making an outcome claim."""
    n = len(window)
    half = max(1, n // 2)
    first = window.iloc[:half]
    second = window.iloc[-half:]

    bullish = float((window["direction"] > 0).mean())
    bearish = float((window["direction"] < 0).mean())
    flips = float((window["direction"].iloc[1:].to_numpy() * window["direction"].iloc[:-1].to_numpy() < 0).mean()) if n > 1 else 0.0
    compression = float((window["range_ratio"] < RANGE_COMPRESSION).mean())
    expansion = float((window["range_ratio"] > RANGE_EXPANSION).mean())
    upper_rejection = float((window["upper_wick_pct"] >= REJECTION_WICK).mean())
    lower_rejection = float((window["lower_wick_pct"] >= REJECTION_WICK).mean())

    first_bull = float((first["direction"] > 0).mean())
    second_bull = float((second["direction"] > 0).mean())
    directional_shift = second_bull - first_bull
    range_shift = float(second["range_ratio"].mean() - first["range_ratio"].mean())
    body_shift = float(second["body_pct"].mean() - first["body_pct"].mean())

    # These are descriptive labels only. They describe the observed sequence
    # geometry; they do not predict what happens after the window.
    if first_bull <= 0.35 and second_bull >= 0.65:
        transition = "BEARISH_TO_BULLISH_SHIFT"
    elif first_bull >= 0.65 and second_bull <= 0.35:
        transition = "BULLISH_TO_BEARISH_SHIFT"
    elif abs(first_bull - 0.5) <= 0.20 and abs(second_bull - 0.5) <= 0.20 and compression >= 0.30:
        transition = "SIDEWAYS_LIKE"
    elif abs(first_bull - 0.5) <= 0.20 and max(second_bull, 1.0 - second_bull) >= 0.65 and range_shift >= 0.15:
        transition = "SIDEWAYS_TO_DIRECTIONAL_LIKE"
    elif range_shift >= 0.20:
        transition = "RANGE_EXPANSION_LIKE"
    elif range_shift <= -0.20:
        transition = "RANGE_COMPRESSION_LIKE"
    else:
        transition = "MIXED_SEQUENCE"

    return {
        "candles": n,
        "bullish_fraction": bullish,
        "bearish_fraction": bearish,
        "alternation_fraction": flips,
        "compression_fraction": compression,
        "expansion_fraction": expansion,
        "upper_rejection_fraction": upper_rejection,
        "lower_rejection_fraction": lower_rejection,
        "mean_body_pct": float(window["body_pct"].mean()),
        "mean_range_ratio": float(window["range_ratio"].mean()),
        "first_half_bullish_fraction": first_bull,
        "second_half_bullish_fraction": second_bull,
        "directional_shift": directional_shift,
        "range_shift": range_shift,
        "body_shift": body_shift,
        "transition_profile": transition,
    }


def _print_sequence(name: str, seq: pd.DataFrame, start: int, end: int) -> None:
    s = _summary(seq)
    print(f"\n{name}: candles {start} -> {end}")
    print(f"Profile: {s['transition_profile']}")
    print(
        "  "
        f"bull={s['bullish_fraction']:.2f} bear={s['bearish_fraction']:.2f} "
        f"alternation={s['alternation_fraction']:.2f} "
        f"compression={s['compression_fraction']:.2f} expansion={s['expansion_fraction']:.2f}"
    )
    print(
        "  "
        f"upper_rejection={s['upper_rejection_fraction']:.2f} "
        f"lower_rejection={s['lower_rejection_fraction']:.2f} "
        f"range_shift={s['range_shift']:+.3f} body_shift={s['body_shift']:+.3f}"
    )
    print("  States:")
    for idx, row in seq.iterrows():
        print(f"    {idx:>6}: {row['state']}")


def _plot_state_matrix(named_windows: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    """Plot compact categorical state matrix for current + historical windows."""
    features = [
        "BULLISH", "BEARISH", "SMALL_BODY", "LARGE_BODY", "COMPRESSION",
        "EXPANSION", "UPPER_REJECTION", "LOWER_REJECTION", "CLOSE_HIGH", "CLOSE_LOW",
    ]
    matrix = np.zeros((len(features), len(named_windows) * len(named_windows[0][1])))
    labels = []
    for col, (name, frame) in enumerate(named_windows):
        for j, (_, row) in enumerate(frame.iterrows()):
            tags = set(row["state"].split("|"))
            for r, feature in enumerate(features):
                matrix[r, col * len(frame) + j] = feature in tags
        labels.extend([f"{name}\n{i+1}" for i in range(len(frame))])

    fig_h = max(5.0, len(features) * 0.42)
    fig_w = max(12.0, len(labels) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_title("Candle Behaviour State Matrix\n(binary presence of descriptive states)")
    ax.set_xlabel("Sequence / candle position")
    ax.set_ylabel("Behaviour state")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive candle behaviour state research")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    db = MarketDatabase(cfg.database)
    raw = db.load_candles(cfg.symbol, args.timeframe)
    raw = raw.reset_index(drop=True)
    if len(raw) < args.window + 5:
        raise ValueError("Not enough candles for requested window")

    features = candle_features(raw)
    states = classify_candles(raw, features)

    current_start = len(raw) - args.window
    current = states.iloc[current_start:].copy()

    ranked = rank_candle_sequences(
        raw,
        window=args.window,
        top_k=args.top_k,
        exclude_recent=args.window,
    )

    print("CANDLE BEHAVIOUR STATE RESEARCH")
    print(f"Symbol: {cfg.symbol} | timeframe: {args.timeframe} | window: {args.window} candles")
    print("Descriptive only: no trades, targets, stops, signals, or outcome predictions.")
    _print_sequence("CURRENT", current, current_start, len(raw) - 1)

    named_windows = [("CURRENT", current)]
    rows = []
    seq_rows = []
    for rank, item in enumerate(ranked[: args.top_k], start=1):
        start = int(item["candidate_start"])
        end = int(item["candidate_end"])
        hist = states.iloc[start : end + 1].copy()
        named_windows.append((f"HIST_{rank}", hist))
        _print_sequence(f"HISTORICAL_{rank} | similarity={item['similarity']:.4f}", hist, start, end)

        summary = _summary(hist)
        rows.append({
            "rank": rank,
            "candidate_start": start,
            "candidate_end": end,
            "similarity": float(item["similarity"]),
            **summary,
        })
        for pos, (idx, row) in enumerate(hist.iterrows(), start=1):
            seq_rows.append({
                "rank": rank,
                "candidate_start": start,
                "candidate_end": end,
                "position": pos,
                "index": int(idx),
                "state": row["state"],
                "body_pct": float(row["body_pct"]),
                "range_ratio": float(row["range_ratio"]),
                "upper_wick_pct": float(row["upper_wick_pct"]),
                "lower_wick_pct": float(row["lower_wick_pct"]),
                "close_location": float(row["close_location"]),
            })

    # Context around the matched windows, useful for studying what happened
    # immediately before/after a similar candle sequence without calling it an outcome.
    context_rows = []
    context_names = [("CURRENT", current)]
    for rank, item in enumerate(ranked[: args.top_k], start=1):
        start = int(item["candidate_start"])
        end = int(item["candidate_end"])
        cstart = max(0, start - args.context)
        cend = min(len(states), end + args.context + 1)
        context = states.iloc[cstart:cend].copy()
        context_names.append((f"HIST_{rank}_CONTEXT", context))
        before = states.iloc[max(0, start - args.context):start]
        after = states.iloc[end + 1:min(len(states), end + 1 + args.context)]
        context_rows.append({
            "rank": rank,
            "similarity": float(item["similarity"]),
            "before_profile": _summary(before)["transition_profile"] if len(before) else "NONE",
            "matched_profile": _summary(states.iloc[start:end + 1])["transition_profile"],
            "after_profile": _summary(after)["transition_profile"] if len(after) else "NONE",
            "before_bullish": float((before["direction"] > 0).mean()) if len(before) else np.nan,
            "matched_bullish": float((states.iloc[start:end + 1]["direction"] > 0).mean()),
            "after_bullish": float((after["direction"] > 0).mean()) if len(after) else np.nan,
            "before_range": float(before["range_ratio"].mean()) if len(before) else np.nan,
            "matched_range": float(states.iloc[start:end + 1]["range_ratio"].mean()),
            "after_range": float(after["range_ratio"].mean()) if len(after) else np.nan,
        })

    charts = Path("charts")
    charts.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(charts / "candle_behavior_state_summary.csv", index=False)
    pd.DataFrame(seq_rows).to_csv(charts / "candle_behavior_state_sequences.csv", index=False)
    pd.DataFrame(context_rows).to_csv(charts / "candle_behavior_state_context.csv", index=False)
    _plot_state_matrix(named_windows, charts / "candle_behavior_state_matrix.png")

    print("\nTRANSITION PROFILE SUMMARY")
    print(pd.DataFrame(rows)[["rank", "similarity", "transition_profile", "bullish_fraction", "bearish_fraction", "compression_fraction", "expansion_fraction", "directional_shift", "range_shift"]].to_string(index=False))
    print("\nSaved:")
    print("  charts/candle_behavior_state_matrix.png")
    print("  charts/candle_behavior_state_summary.csv")
    print("  charts/candle_behavior_state_sequences.csv")
    print("  charts/candle_behavior_state_context.csv")


if __name__ == "__main__":
    main()
