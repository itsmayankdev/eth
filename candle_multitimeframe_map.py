"""Multi-timeframe candle-behaviour relationship research.

Descriptive research only. Aligns the current candle-behaviour fingerprints and
transition descriptions across timeframes. It does not create trades, targets,
stops, outcomes, or future-direction predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import FEATURES, candle_features
from data.database import MarketDatabase

TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_WINDOW = 20


def window_summary(features: pd.DataFrame, window: int) -> dict:
    v = features[list(FEATURES)].to_numpy(float)[-window:]
    idx = {n: FEATURES.index(n) for n in FEATURES}
    direction = v[:, idx["direction"]]
    body = v[:, idx["body_pct"]]
    rng = v[:, idx["range_ratio"]]
    upper = v[:, idx["upper_wick_pct"]]
    lower = v[:, idx["lower_wick_pct"]]
    close = v[:, idx["close_location"]]
    alt = v[:, idx["alternation"]]
    half = window // 2

    d1, d2 = direction[:half].mean(), direction[half:].mean()
    r1, r2 = rng[:half].mean(), rng[half:].mean()
    b1, b2 = body[:half].mean(), body[half:].mean()
    c1, c2 = close[:half].mean(), close[half:].mean()

    if abs(d1) >= .20 and abs(d2) >= .15 and d1 * d2 < 0:
        regime = "OPPOSITE_DIRECTION_SHIFT"
    elif abs(d1) <= .15 and abs(d2) >= .20 and r2 - r1 >= .15:
        regime = "SIDEWAYS_TO_DIRECTIONAL"
    elif abs(d1) >= .20 and abs(d2) <= .15:
        regime = "DIRECTIONAL_TO_SIDEWAYS"
    elif r2 - r1 >= .25:
        regime = "COMPRESSION_TO_EXPANSION"
    elif r2 - r1 <= -.25:
        regime = "EXPANSION_TO_COMPRESSION"
    elif abs(d1-d2) < .15 and abs(r1-r2) < .15:
        regime = "STABLE"
    else:
        regime = "MIXED"

    return {
        "direction_balance": float(direction.mean()),
        "first_direction": float(d1), "second_direction": float(d2),
        "body_level": float(body.mean()), "body_shift": float(b2-b1),
        "range_level": float(rng.mean()), "range_shift": float(r2-r1),
        "upper_wick": float(upper.mean()), "lower_wick": float(lower.mean()),
        "close_location": float(close.mean()), "close_shift": float(c2-c1),
        "alternation": float(alt.mean()), "transition_type": regime,
    }


def normalized_relationship(rows: list[dict]) -> str:
    dirs = np.array([r["direction_balance"] for r in rows])
    ranges = np.array([r["range_level"] for r in rows])
    shifts = np.array([r["range_shift"] for r in rows])
    labels = [r["transition_type"] for r in rows]
    if len(dirs) >= 2:
        spread = float(dirs.max() - dirs.min())
        range_spread = float(ranges.max() - ranges.min())
    else:
        spread = range_spread = 0.0
    if spread <= .20:
        direction_relation = "DIRECTION_ALIGNED"
    elif spread >= .50:
        direction_relation = "DIRECTION_DIVERGENT"
    else:
        direction_relation = "DIRECTION_MIXED"
    if range_spread <= .25:
        range_relation = "RANGE_ALIGNED"
    elif range_spread >= .75:
        range_relation = "RANGE_DIVERGENT"
    else:
        range_relation = "RANGE_MIXED"
    if any("SIDEWAYS_TO_DIRECTIONAL" in x or "COMPRESSION_TO_EXPANSION" in x for x in labels):
        transition_relation = "EXPANSION_CONTEXT_PRESENT"
    elif any("DIRECTIONAL_TO_SIDEWAYS" in x or "EXPANSION_TO_COMPRESSION" in x for x in labels):
        transition_relation = "COOLING_CONTEXT_PRESENT"
    elif any("OPPOSITE_DIRECTION_SHIFT" in x for x in labels):
        transition_relation = "DIRECTION_SHIFT_PRESENT"
    else:
        transition_relation = "NO_MAJOR_CROSS_TF_SHIFT"
    return f"{direction_relation} | {range_relation} | {transition_relation}"


def run(window: int, output_dir: str):
    if window < 4 or window % 2:
        raise ValueError("window must be an even number >= 4")
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    for name in ["candle_multitimeframe_current.csv", "candle_multitimeframe_relationships.csv", "candle_multitimeframe_heatmap.png"]:
        p = out / name
        if p.exists():
            p.unlink()

    db = MarketDatabase("data/eth_market.db")
    rows = []
    for tf in TIMEFRAMES:
        df = db.load_candles("ETHUSDT", tf)
        if df.empty or len(df) < window:
            continue
        summary = window_summary(candle_features(df), window)
        summary["timeframe"] = tf
        rows.append(summary)
        print(f"[{tf}] {summary['transition_type']} | dir={summary['direction_balance']:.2f} | range={summary['range_level']:.2f}")

    current = pd.DataFrame(rows).set_index("timeframe").loc[[x for x in TIMEFRAMES if x in pd.DataFrame(rows).set_index("timeframe").index]].reset_index()
    relationship = normalized_relationship(rows)
    current["cross_timeframe_relationship"] = relationship
    current.to_csv(out / "candle_multitimeframe_current.csv", index=False)

    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            pairs.append({
                "lower_timeframe": a["timeframe"], "higher_timeframe": b["timeframe"],
                "direction_difference": float(a["direction_balance"] - b["direction_balance"]),
                "range_difference": float(a["range_level"] - b["range_level"]),
                "range_shift_difference": float(a["range_shift"] - b["range_shift"]),
                "lower_transition": a["transition_type"], "higher_transition": b["transition_type"],
            })
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(out / "candle_multitimeframe_relationships.csv", index=False)

    matrix = current.set_index("timeframe")["direction_balance"].to_numpy(dtype=float).reshape(-1, 1)
    plt.figure(figsize=(5, 7))
    plt.imshow(matrix, aspect="auto")
    plt.yticks(range(len(current)), current["timeframe"])
    plt.xticks([0], ["Direction balance"])
    plt.colorbar(label="Direction balance")
    plt.title("Current Multi-Timeframe Candle Behaviour")
    plt.tight_layout()
    plt.savefig(out / "candle_multitimeframe_heatmap.png", dpi=150)
    plt.close()

    print("MULTI-TIMEFRAME CANDLE BEHAVIOUR RESEARCH")
    print(current[["timeframe", "direction_balance", "body_level", "range_level", "range_shift", "close_location", "transition_type"]].to_string(index=False))
    print(f"Cross-timeframe relationship: {relationship}")
    print("Saved multi-timeframe research outputs in charts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args()
    run(args.window, args.output_dir)
