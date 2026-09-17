"""Historical multi-timeframe candle-behaviour relationship research.

Descriptive research only. Finds historical periods whose candle behaviour,
measured across all available timeframes over the same historical time window,
resembles the current multi-timeframe configuration. It does not create trades,
targets, stops, outcomes, or future-direction predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from candle_similarity import FEATURES, candle_features
from data.database import MarketDatabase

TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
DEFAULT_STEP = 20
ANCHOR_TF = "1h"

TF_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def feature_vector(features: pd.DataFrame, start: int, window: int) -> np.ndarray:
    v = features[list(FEATURES)].to_numpy(float)[start:start + window]
    idx = {n: i for i, n in enumerate(FEATURES)}
    d = v[:, idx["direction"]]
    body = v[:, idx["body_pct"]]
    rng = v[:, idx["range_ratio"]]
    upper = v[:, idx["upper_wick_pct"]]
    lower = v[:, idx["lower_wick_pct"]]
    close = v[:, idx["close_location"]]
    alt = v[:, idx["alternation"]]
    h = window // 2
    return np.array([
        d.mean(), body.mean(), rng.mean(), upper.mean(), lower.mean(),
        close.mean(), alt.mean(),
        d[:h].mean(), d[h:].mean(),
        rng[:h].mean(), rng[h:].mean(),
        body[:h].mean(), body[h:].mean(),
        close[:h].mean(), close[h:].mean(),
        alt[:h].mean(), alt[h:].mean(),
        rng[h:].mean() - rng[:h].mean(),
        d[h:].mean() - d[:h].mean(),
    ], dtype=float)


def build_history(features: pd.DataFrame, window: int, step: int):
    n = len(features)
    starts = np.arange(0, n - window + 1, step, dtype=int)
    matrix = np.vstack([feature_vector(features, int(s), window) for s in starts])
    return matrix, starts


def robust_scale(matrix: np.ndarray):
    med = np.nanmedian(matrix, axis=0)
    scale = np.nanmedian(np.abs(matrix - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return med, scale


def similarity(current: np.ndarray, history: np.ndarray) -> np.ndarray:
    med, scale = robust_scale(history)
    z_hist = (history - med) / scale
    z_current = (current - med) / scale
    dist = np.sqrt(np.mean((z_hist - z_current[None, :]) ** 2, axis=1))
    return np.exp(-dist)


def floor_index(times: np.ndarray, target: int) -> int | None:
    pos = int(np.searchsorted(times, target, side="right")) - 1
    return pos if pos >= 0 else None


def aligned_index(times: np.ndarray, target: int, timeframe: str) -> int | None:
    """Align to the latest available window endpoint at/before the anchor."""
    pos = floor_index(times, target)
    if pos is None:
        return None
    lag = int(target) - int(times[pos])
    if lag > int(TF_MS[timeframe] * 1.5):
        return None
    return pos


def run(window: int, top_k: int, step: int, output_dir: str):
    if window < 4 or window % 2:
        raise ValueError("window must be an even number >= 4")
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    db = MarketDatabase("data/eth_market.db")
    data = {}

    for tf in TIMEFRAMES:
        df = db.load_candles("ETHUSDT", tf)
        if len(df) < window:
            continue
        df = df.sort_values("open_time").reset_index(drop=True)
        feats = candle_features(df)
        matrix, starts = build_history(feats, window, step)
        current = feature_vector(feats, len(feats) - window, window)
        sims = similarity(current, matrix)
        ends = starts + window - 1
        times = df.iloc[ends]["open_time"].to_numpy(dtype=np.int64)
        data[tf] = {
            "df": df,
            "matrix": matrix,
            "starts": starts,
            "ends": ends,
            "times": times,
            "current": current,
            "similarity": sims,
        }
        print(f"[{tf}] {len(matrix)} historical windows prepared")

    if ANCHOR_TF not in data:
        raise RuntimeError("1h data is required as the historical alignment anchor")

    current_rows = []
    for tf in TIMEFRAMES:
        if tf not in data:
            continue
        x = data[tf]["current"]
        current_rows.append({
            "timeframe": tf,
            "direction_balance": x[0],
            "body_level": x[1],
            "range_level": x[2],
            "upper_wick": x[3],
            "lower_wick": x[4],
            "close_location": x[5],
            "alternation": x[6],
            "direction_shift": x[18],
            "range_shift": x[17],
        })
    current_df = pd.DataFrame(current_rows)

    # Find the time period that exists in EVERY timeframe's historical windows.
    # This is critical because the snapshots have different history lengths:
    # 1m is much shorter than 1d. Candidate anchors outside this overlap cannot
    # possibly produce a true all-timeframe historical comparison.
    common_start = max(int(item["times"][0]) for item in data.values())
    common_end = min(int(item["times"][-1]) for item in data.values())

    if common_start > common_end:
        raise RuntimeError(
            "No common historical overlap across all timeframes: "
            f"start={common_start}, end={common_end}"
        )

    anchor = data[ANCHOR_TF]
    current_anchor_start = len(anchor["df"]) - window
    overlap_mask = (
        (anchor["times"] >= common_start)
        & (anchor["times"] <= common_end)
        & (anchor["starts"] != current_anchor_start)
    )
    candidate_indices = np.flatnonzero(overlap_mask)

    print(f"Common historical overlap: {common_start} -> {common_end}")
    print(f"Synchronized 1h anchor candidates: {len(candidate_indices)}")

    if len(candidate_indices) == 0:
        raise RuntimeError("No synchronized historical windows in the common timeframe overlap")

    # Rank ONLY candidates inside the common overlap.
    ranked = candidate_indices[np.argsort(anchor["similarity"][candidate_indices])[::-1]]

    results = []
    skipped = 0
    for rank, i in enumerate(ranked, 1):
        anchor_ts = int(anchor["times"][i])
        row = {
            "anchor_rank_1h": rank,
            "anchor_timestamp_1h": anchor_ts,
        }
        scores = []
        aligned_ok = True

        for tf in TIMEFRAMES:
            if tf not in data:
                aligned_ok = False
                break
            item = data[tf]
            j = aligned_index(item["times"], anchor_ts, tf)
            if j is None:
                aligned_ok = False
                break
            row[f"{tf}_similarity"] = float(item["similarity"][j])
            row[f"{tf}_timestamp"] = int(item["times"][j])
            scores.append(float(item["similarity"][j]))

        if not aligned_ok or len(scores) != len(TIMEFRAMES):
            skipped += 1
            continue

        row["combined_similarity"] = float(np.mean(scores))
        row["max_alignment_gap_ms"] = max(
            anchor_ts - row[f"{tf}_timestamp"] for tf in TIMEFRAMES
        )
        row["alignment_status"] = "SYNCHRONIZED"
        results.append(row)

        if len(results) >= max(top_k, 25):
            break

    if not results:
        raise RuntimeError("No synchronized historical multi-timeframe windows found after alignment")

    matches = (
        pd.DataFrame(results)
        .sort_values("combined_similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    current_df["research_note"] = "Descriptive candle-behaviour comparison only"
    matches["research_note"] = "Synchronized historical multi-timeframe similarity; not a forecast"

    current_df.to_csv(out / "candle_mtf_history_current.csv", index=False)
    matches.to_csv(out / "candle_mtf_history_matches.csv", index=False)

    print("\nMULTI-TIMEFRAME SYNCHRONIZED HISTORICAL CANDLE BEHAVIOUR RESEARCH\n")
    print(current_df.to_string(index=False))
    print("\nTOP SYNCHRONIZED HISTORICAL MULTI-TIMEFRAME WINDOWS\n")
    cols = [
        "anchor_rank_1h",
        "anchor_timestamp_1h",
        "combined_similarity",
        "max_alignment_gap_ms",
    ]
    cols += [f"{tf}_similarity" for tf in TIMEFRAMES]
    print(matches[cols].to_string(index=False))
    print(f"\nCandidates skipped during final alignment: {skipped}")
    print("\nSaved:")
    print(out / "candle_mtf_history_current.csv")
    print(out / "candle_mtf_history_matches.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args()
    run(args.window, args.top_k, args.step, args.output_dir)
