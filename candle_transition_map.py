"""Candle-behaviour transition and regime research.

Descriptive research only. This module identifies observed behaviour changes inside
historical candle windows: trend-to-opposite-direction shifts, compression/sideways
to directional expansion, and persistent regimes. It does not create trades,
targets, stops, outcomes, or future-direction predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import FEATURES, CandleSimilarityConfig, candle_features
from data.database import MarketDatabase

TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 15
SAMPLE_STEP = 10
CHUNK_SIZE = 4096


def rolling_view(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.empty((0, window, values.shape[1]), dtype=float)
    return np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)


def robust_window_features(features: pd.DataFrame, window: int, step: int):
    values = features[list(FEATURES)].to_numpy(float)
    views = rolling_view(values, window)[::step]
    ends = np.arange(window - 1, len(values), step, dtype=int)
    if len(views) == 0:
        return np.empty((0, 0)), ends

    idx = {name: FEATURES.index(name) for name in FEATURES}
    direction = views[:, :, idx["direction"]]
    body = views[:, :, idx["body_pct"]]
    rng = views[:, :, idx["range_ratio"]]
    upper = views[:, :, idx["upper_wick_pct"]]
    lower = views[:, :, idx["lower_wick_pct"]]
    close = views[:, :, idx["close_location"]]
    alt = views[:, :, idx["alternation"]]

    half = window // 2
    a = slice(0, half)
    b = slice(half, window)

    # Window-level anatomy.
    pieces = [
        direction.mean(1),
        body.mean(1),
        rng.mean(1),
        upper.mean(1),
        lower.mean(1),
        close.mean(1),
        alt.mean(1),
        direction[:, a].mean(1),
        direction[:, b].mean(1),
        body[:, a].mean(1),
        body[:, b].mean(1),
        rng[:, a].mean(1),
        rng[:, b].mean(1),
        close[:, a].mean(1),
        close[:, b].mean(1),
        alt[:, a].mean(1),
        alt[:, b].mean(1),
        (rng[:, b].mean(1) - rng[:, a].mean(1)),
        (body[:, b].mean(1) - body[:, a].mean(1)),
        (direction[:, b].mean(1) - direction[:, a].mean(1)),
        (close[:, b].mean(1) - close[:, a].mean(1)),
        (upper[:, b].mean(1) - upper[:, a].mean(1)),
        (lower[:, b].mean(1) - lower[:, a].mean(1)),
    ]
    return np.column_stack(pieces), ends


def transition_label(row: np.ndarray) -> str:
    """Assign a descriptive label to the observed first-half -> second-half change."""
    d1, d2 = row[7], row[8]
    r1, r2 = row[11], row[12]
    a1, a2 = row[15], row[16]
    direction_shift = row[19]
    range_shift = row[17]

    if abs(d1) >= 0.20 and abs(d2) >= 0.15 and d1 * d2 < 0 and direction_shift * d1 < 0:
        return "TREND_TO_OPPOSITE_DIRECTION"
    if abs(d1) <= 0.15 and abs(d2) >= 0.20 and range_shift >= 0.15:
        return "SIDEWAYS_TO_DIRECTIONAL_EXPANSION"
    if abs(d1) >= 0.20 and abs(d2) <= 0.15 and range_shift < 0.10:
        return "DIRECTIONAL_TO_SIDEWAYS"
    if range_shift >= 0.25 and abs(d2) >= abs(d1) + 0.10:
        return "COMPRESSION_TO_EXPANSION"
    if range_shift <= -0.25 and abs(d2) <= abs(d1) + 0.05:
        return "EXPANSION_TO_COMPRESSION"
    if abs(d1) <= 0.15 and abs(d2) <= 0.15 and a2 >= a1 + 0.10:
        return "SIDEWAYS_ALTERNATION_INCREASE"
    if abs(d1 - d2) < 0.15 and abs(r1 - r2) < 0.15:
        return "STABLE_REGIME"
    return "MIXED_TRANSITION"


def transition_similarity(current: np.ndarray, candidates: np.ndarray, top_k: int):
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    # Robust feature scaling prevents one measurement from dominating.
    med = np.nanmedian(candidates, axis=0)
    scale = np.nanmedian(np.abs(candidates - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    c = (candidates - med) / scale
    q = (current - med) / scale
    weights = np.ones(c.shape[1], dtype=float)
    # Give first/second-half direction and range changes slightly more weight.
    weights[[7, 8, 11, 12, 17, 19]] = 1.5
    weights /= weights.sum()
    distance = np.sqrt(np.mean((c - q[None, :]) ** 2, axis=1))
    score = np.exp(-distance * (1.0 / max(weights.mean(), 1e-9)))
    take = min(top_k, len(score))
    idx = np.argpartition(score, -take)[-take:]
    return idx[np.argsort(score[idx])[::-1]], score[idx[np.argsort(score[idx])[::-1]]]


def describe_row(row: np.ndarray) -> dict:
    labels = {
        "first_direction_balance": float(row[7]),
        "second_direction_balance": float(row[8]),
        "first_range_level": float(row[11]),
        "second_range_level": float(row[12]),
        "first_alternation": float(row[15]),
        "second_alternation": float(row[16]),
        "range_shift": float(row[17]),
        "body_shift": float(row[18]),
        "direction_shift": float(row[19]),
        "close_shift": float(row[20]),
        "upper_wick_shift": float(row[21]),
        "lower_wick_shift": float(row[22]),
    }
    labels["transition_type"] = transition_label(row)
    return labels


def analyze_timeframe(db, timeframe: str, window: int, top_k: int):
    df = db.load_candles("ETHUSDT", timeframe)
    if df.empty or len(df) < window * 2:
        return None
    features = candle_features(df)
    matrix, ends = robust_window_features(features, window, SAMPLE_STEP)
    current_start = len(features) - window
    current_matrix, _ = robust_window_features(features.iloc[current_start:], window, 1)
    if len(current_matrix) == 0:
        return None
    current = current_matrix[0]

    mask = ends < current_start
    hist = matrix[mask]
    hist_ends = ends[mask]
    labels = np.array([transition_label(row) for row in hist])
    current_label = transition_label(current)

    # Match only within the same descriptive transition regime when possible.
    same = labels == current_label
    pool = hist[same] if same.sum() >= 3 else hist
    pool_ends = hist_ends[same] if same.sum() >= 3 else hist_ends
    idx, scores = transition_similarity(current, pool, top_k)

    matches = []
    for rank, (i, score) in enumerate(zip(idx, scores), 1):
        end = int(pool_ends[i])
        row = pool[i]
        item = {
            "timeframe": timeframe,
            "rank": rank,
            "candidate_end": end,
            "candidate_start": end - window + 1,
            "similarity": float(score),
            **describe_row(row),
        }
        matches.append(item)

    current_info = {
        "timeframe": timeframe,
        "window": window,
        "current_transition_type": current_label,
        "historical_windows": len(hist),
        "same_regime_windows": int(same.sum()),
        **describe_row(current),
    }
    counts = pd.Series(labels).value_counts().rename_axis("transition_type").reset_index(name="historical_windows")
    counts.insert(0, "timeframe", timeframe)
    counts["share_pct"] = counts["historical_windows"] / len(hist) * 100
    return current_info, pd.DataFrame(matches), counts


def run(all_timeframes: bool, timeframe: str, window: int, top_k: int):
    if window % 2:
        raise ValueError("window must be even so the transition map can compare two equal halves")
    out = Path("charts")
    out.mkdir(exist_ok=True)
    for name in [
        "candle_transition_current.csv",
        "candle_transition_matches.csv",
        "candle_transition_distribution.csv",
        "candle_transition_map.png",
    ]:
        p = out / name
        if p.exists():
            p.unlink()

    db = MarketDatabase("data/eth_market.db")
    tfs = TIMEFRAMES if all_timeframes else (timeframe,)
    currents, matches, distributions = [], [], []

    for tf in tfs:
        print(f"[{tf}] building transition/regime map...")
        result = analyze_timeframe(db, tf, window, top_k)
        if result is None:
            print(f"[{tf}] insufficient data")
            continue
        current, match_df, dist_df = result
        currents.append(current)
        if not match_df.empty:
            matches.append(match_df)
        distributions.append(dist_df)
        print(
            f"[{tf}] {current['current_transition_type']} | "
            f"first dir={current['first_direction_balance']:.2f} -> "
            f"second dir={current['second_direction_balance']:.2f} | "
            f"range shift={current['range_shift']:.2f}"
        )

    current_df = pd.DataFrame(currents)
    matches_df = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    dist_df = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame()
    current_df.to_csv(out / "candle_transition_current.csv", index=False)
    matches_df.to_csv(out / "candle_transition_matches.csv", index=False)
    dist_df.to_csv(out / "candle_transition_distribution.csv", index=False)

    if not current_df.empty:
        plt.figure(figsize=(12, 5))
        counts = current_df["current_transition_type"].value_counts()
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Current timeframe count")
        plt.title("Current Candle Behaviour Transition Regimes")
        plt.tight_layout()
        plt.savefig(out / "candle_transition_map.png", dpi=150)
        plt.close()

    print("CANDLE BEHAVIOUR TRANSITION / REGIME RESEARCH")
    print(current_df[["timeframe", "current_transition_type", "first_direction_balance", "second_direction_balance", "range_shift", "body_shift", "close_shift"]].to_string(index=False))
    print("Saved transition research outputs in charts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="1h")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--all-timeframes", action="store_true")
    args = parser.parse_args()
    run(args.all_timeframes, args.timeframe, args.window, args.top_k)
