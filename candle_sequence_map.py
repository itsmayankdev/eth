"""Sequential candle-behaviour map research.

Breaks each candle window into ordered blocks and describes how body, range,
direction, wick and close-location behaviour changes from block to block.
Research-only: no signals, trades, targets, stops, outcomes, or predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import candle_features, rank_candle_sequences
from config import load_config
from data.database import MarketDatabase

RANGE_SCALE = 5.0
BLOCKS = 5


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        raw = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if raw.empty:
        raise ValueError(f"No data available for {timeframe}")
    return raw.reset_index(drop=True)


def rr(f):
    return f["range_ratio"] * RANGE_SCALE


def frac(mask):
    return float(mask.mean()) if len(mask) else 0.0


def block_features(f: pd.DataFrame) -> dict:
    direction = f.direction.to_numpy(dtype=float)
    r = rr(f)
    body = f.body_pct
    close = f.close_location
    bull = frac(f.direction > 0)
    alt = frac((direction[1:] * direction[:-1]) < 0) if len(f) > 1 else 0.0
    upper = frac(f.upper_wick_pct >= 0.35)
    lower = frac(f.lower_wick_pct >= 0.35)
    return {
        "bull": bull,
        "bear": frac(f.direction < 0),
        "alt": alt,
        "body": float(body.mean()),
        "range": float(r.mean()),
        "upper": upper,
        "lower": lower,
        "close": float(close.mean()),
        "compression": frac(r < 0.85),
        "expansion": frac(r > 1.15),
    }


def split_blocks(f: pd.DataFrame, n_blocks=BLOCKS):
    return [f.iloc[idx] for idx in np.array_split(np.arange(len(f)), n_blocks) if len(idx)]


def classify_block(x: dict) -> str:
    direction = "BULL" if x["bull"] >= 0.60 else ("BEAR" if x["bull"] <= 0.40 else "BAL")
    range_state = "EXP" if x["range"] >= 1.15 else ("COMP" if x["range"] <= 0.85 else "NORM")
    wick = "LOWER_WICK" if x["lower"] - x["upper"] >= 0.15 else ("UPPER_WICK" if x["upper"] - x["lower"] >= 0.15 else "WICK_BAL")
    alt = "ALT_HI" if x["alt"] >= 0.55 else ("ALT_LO" if x["alt"] <= 0.30 else "ALT_MID")
    body = "BODY_LARGE" if x["body"] >= 0.60 else ("BODY_SMALL" if x["body"] <= 0.25 else "BODY_MID")
    close = "CLOSE_HIGH" if x["close"] >= 0.65 else ("CLOSE_LOW" if x["close"] <= 0.35 else "CLOSE_MID")
    return f"{direction}/{range_state}/{wick}/{alt}/{body}/{close}"


def sequence_map(f: pd.DataFrame, n_blocks=BLOCKS) -> tuple[list[dict], dict]:
    blocks = split_blocks(f, n_blocks)
    rows = []
    for i, b in enumerate(blocks, 1):
        x = block_features(b)
        rows.append({"block": i, **x, "label": classify_block(x)})

    first = rows[0]
    last = rows[-1]
    summary = {
        "block_count": len(rows),
        "block_direction_path": " -> ".join(x["label"].split("/")[0] for x in rows),
        "block_range_path": " -> ".join(x["label"].split("/")[1] for x in rows),
        "block_wick_path": " -> ".join(x["label"].split("/")[2] for x in rows),
        "block_alternation_path": " -> ".join(x["label"].split("/")[3] for x in rows),
        "block_body_path": " -> ".join(x["label"].split("/")[4] for x in rows),
        "block_close_path": " -> ".join(x["label"].split("/")[5] for x in rows),
        "direction_change": last["bull"] - first["bull"],
        "range_change": last["range"] - first["range"],
        "body_change": last["body"] - first["body"],
        "wick_change": (last["lower"] - last["upper"]) - (first["lower"] - first["upper"]),
        "close_change": last["close"] - first["close"],
    }
    return rows, summary


def rank_windows(features, current_end, window, top_k):
    """Fast historical ranking; detailed block mapping is only done for top-K."""
    current_start = current_end - window + 1
    candidate_ends = range(window - 1, current_start)
    ranked = rank_candle_sequences(
        features,
        current_start=current_start,
        current_end=current_end,
        candidate_ends=candidate_ends,
        chunk_size=4096,
    )
    return ranked.head(top_k).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--all-timeframes", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    timeframes = cfg.timeframes if args.all_timeframes else [args.timeframe]
    charts = Path("charts")
    charts.mkdir(exist_ok=True)
    for name in ["candle_sequence_current.csv", "candle_sequence_matches.csv", "candle_sequence_blocks.csv", "candle_sequence_map.png"]:
        path = charts / name
        if path.exists():
            path.unlink()

    current_rows, match_rows, block_rows = [], [], []
    for tf in timeframes:
        print(f"[{tf}] loading and vectorizing historical windows...", flush=True)
        raw = load_market(cfg, tf)
        if len(raw) < args.window * 2:
            continue
        f = candle_features(raw)
        current_end = len(f) - 1
        current = f.iloc[current_end - args.window + 1:current_end + 1]
        blocks, summary = sequence_map(current)
        current_rows.append({"timeframe": tf, **summary, "sequence_signature": " || ".join(x["label"] for x in blocks)})
        for b in blocks:
            block_rows.append({"timeframe": tf, "source": "current", "rank": 0, **b})

        ranked = rank_windows(f, current_end, args.window, args.top_k)
        print(f"[{tf}] ranked {len(ranked)} top historical windows", flush=True)
        for rank, row in ranked.iterrows():
            s, e = int(row.candidate_start), int(row.candidate_end)
            match = f.iloc[s:e + 1]
            blocks, summary = sequence_map(match)
            match_rows.append({"timeframe": tf, "rank": rank + 1, "candidate_start": s, "candidate_end": e, "similarity": float(row.similarity), **summary, "sequence_signature": " || ".join(x["label"] for x in blocks)})
            for b in blocks:
                block_rows.append({"timeframe": tf, "source": "historical", "rank": rank + 1, "candidate_start": s, "candidate_end": e, "similarity": float(row.similarity), **b})

    current_df = pd.DataFrame(current_rows)
    matches_df = pd.DataFrame(match_rows)
    blocks_df = pd.DataFrame(block_rows)
    current_df.to_csv(charts / "candle_sequence_current.csv", index=False)
    matches_df.to_csv(charts / "candle_sequence_matches.csv", index=False)
    blocks_df.to_csv(charts / "candle_sequence_blocks.csv", index=False)

    if not matches_df.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        for tf, g in matches_df[matches_df["rank"] <= min(10, args.top_k)].groupby("timeframe"):
            ax.plot(g["rank"], g["similarity"], marker="o", label=tf)
        ax.set_title("Sequential Candle Behaviour — Historical Similarity")
        ax.set_xlabel("Historical neighbour rank")
        ax.set_ylabel("Candle similarity")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "candle_sequence_map.png", dpi=160)
        plt.close(fig)

    print("CANDLE SEQUENCE MAP RESEARCH")
    print("Five ordered blocks per window. Descriptive only; no signals, trades, targets, stops, outcomes, or predictions.")
    print("\nCURRENT SEQUENCE MAPS")
    if not current_df.empty:
        print(current_df[["timeframe", "block_direction_path", "block_range_path", "block_wick_path", "block_alternation_path", "block_body_path", "block_close_path", "direction_change", "range_change", "body_change", "wick_change", "close_change"]].to_string(index=False))
    print("\nTOP HISTORICAL SEQUENCE MAPS")
    if not matches_df.empty:
        print(matches_df[["timeframe", "rank", "similarity", "block_direction_path", "block_range_path", "block_wick_path", "block_body_path", "block_close_path"]].head(80).to_string(index=False))
    print("\nSaved sequence-map outputs in charts/")


if __name__ == "__main__":
    main()
