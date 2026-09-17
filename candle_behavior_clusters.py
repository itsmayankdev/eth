"""Continuous candle-behaviour clustering and rarity research.

Research-only descriptive layer. It groups historical OHLC windows by their
continuous candle anatomy/behaviour and measures how unusual the current
window is relative to historical windows. No signals, trades, targets, stops,
outcomes, or future-direction predictions are calculated.
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
BLOCKS = 5
RANGE_SCALE = 5.0
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
DEFAULT_CLUSTERS = 8
CHUNK_SIZE = 4096


def _rolling_view(values: np.ndarray, window: int) -> np.ndarray:
    """Return rolling windows as a zero-copy view: [n_windows, window, features]."""
    if len(values) < window:
        return np.empty((0, window, values.shape[1]), dtype=values.dtype)
    return np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window, axis=0
    ).transpose(0, 2, 1)


def _standardize_batch(windows: np.ndarray) -> np.ndarray:
    med = np.nanmedian(windows, axis=1, keepdims=True)
    scale = np.nanmedian(np.abs(windows - med), axis=1, keepdims=True) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (windows - med) / scale


def _weighted_similarity_batch(current: np.ndarray, candidates: np.ndarray,
                               weights: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of candle_sequence_similarity for many windows."""
    cur = _standardize_batch(current[None, ...])[0]
    cand = _standardize_batch(candidates)
    per_feature = np.sqrt(np.mean((cand - cur[None, ...]) ** 2, axis=1))
    distance = np.sum(per_feature * weights[None, :], axis=1)
    return np.exp(-distance)


def rolling_similarity(features: pd.DataFrame, window: int, top_k: int,
                       config: CandleSimilarityConfig | None = None) -> pd.DataFrame:
    """Rank non-overlapping historical windows against the current window."""
    config = config or CandleSimilarityConfig()
    values = features[list(FEATURES)].to_numpy(dtype=float)
    views = _rolling_view(values, window)
    if len(views) < 2:
        return pd.DataFrame(columns=["rank", "candidate_start", "candidate_end", "similarity"])

    current_end = len(values) - 1
    current_start = current_end - window + 1
    last_allowed_end = current_start - 1
    candidate_count = last_allowed_end - window + 2
    if candidate_count <= 0:
        return pd.DataFrame(columns=["rank", "candidate_start", "candidate_end", "similarity"])

    weights = config.weights
    weights = weights / weights.sum()
    # Keep only a small working top-k buffer instead of retaining all scores.
    best_scores = np.full(top_k, -np.inf, dtype=float)
    best_ends = np.full(top_k, -1, dtype=int)

    for start in range(0, candidate_count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, candidate_count)
        scores = _weighted_similarity_batch(
            values[current_start:current_end + 1], views[start:stop], weights
        )
        take = min(top_k, len(scores))
        if take == 0:
            continue
        idx = np.argpartition(scores, -take)[-take:]
        combined_scores = np.concatenate([best_scores, scores[idx]])
        combined_ends = np.concatenate([best_ends, np.arange(start, stop)[idx]])
        keep = np.argpartition(combined_scores, -top_k)[-top_k:]
        best_scores = combined_scores[keep]
        best_ends = combined_ends[keep]

    order = np.argsort(best_scores)[::-1]
    best_scores = best_scores[order]
    best_ends = best_ends[order]
    valid = best_ends >= 0
    best_ends = best_ends[valid]
    best_scores = best_scores[valid]
    out = pd.DataFrame({
        "candidate_start": best_ends,
        "candidate_end": best_ends + window - 1,
        "similarity": best_scores,
    })
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def continuous_fingerprint(features: pd.DataFrame) -> np.ndarray:
    """Build a compact continuous vector from one candle window."""
    x = features[list(FEATURES)].to_numpy(dtype=float)
    n = len(x)
    thirds = np.array_split(np.arange(n), 3)
    blocks = np.array_split(np.arange(n), BLOCKS)

    def mean_col(indices, col):
        return float(np.nanmean(x[indices, col])) if len(indices) else 0.0

    direction = x[:, FEATURES.index("direction")]
    body = x[:, FEATURES.index("body_pct")]
    rng = x[:, FEATURES.index("range_ratio")]
    upper = x[:, FEATURES.index("upper_wick_pct")]
    lower = x[:, FEATURES.index("lower_wick_pct")]
    close = x[:, FEATURES.index("close_location")]
    alt = x[:, FEATURES.index("alternation")]

    vector = [
        float(np.mean(direction)),
        float(np.mean(alt)),
        float(np.mean(body)),
        float(np.mean(rng)),
        float(np.mean(upper)),
        float(np.mean(lower)),
        float(np.mean(close)),
        float(np.mean(body <= 0.25)),
        float(np.mean(body >= 0.60)),
        float(np.mean(rng <= 0.85)),
        float(np.mean(rng >= 1.15)),
        float(np.mean(close >= 0.65)),
        float(np.mean(close <= 0.35)),
        float(np.mean(direction > 0)),
        float(np.mean(direction < 0)),
    ]

    for idx in thirds:
        vector.extend([
            mean_col(idx, FEATURES.index("direction")),
            mean_col(idx, FEATURES.index("body_pct")),
            mean_col(idx, FEATURES.index("range_ratio")),
            mean_col(idx, FEATURES.index("close_location")),
            mean_col(idx, FEATURES.index("alternation")),
        ])

    for idx in blocks:
        vector.extend([
            mean_col(idx, FEATURES.index("direction")),
            mean_col(idx, FEATURES.index("body_pct")),
            mean_col(idx, FEATURES.index("range_ratio")),
            mean_col(idx, FEATURES.index("upper_wick_pct")),
            mean_col(idx, FEATURES.index("lower_wick_pct")),
            mean_col(idx, FEATURES.index("close_location")),
            mean_col(idx, FEATURES.index("alternation")),
        ])
    return np.asarray(vector, dtype=float)


def fingerprint_matrix(features: pd.DataFrame, window: int, step: int = 1):
    """Create continuous fingerprints for rolling historical windows."""
    n = len(features)
    if n < window:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    # Fingerprints are intentionally built in chunks to keep memory bounded.
    rows = []
    ends = []
    for end in range(window - 1, n, step):
        start = end - window + 1
        rows.append(continuous_fingerprint(features.iloc[start:end + 1]))
        ends.append(end)
    return np.vstack(rows), np.asarray(ends, dtype=int)


def standardize_rows(matrix: np.ndarray) -> np.ndarray:
    med = np.nanmedian(matrix, axis=0)
    scale = np.nanmedian(np.abs(matrix - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (matrix - med) / scale


def kmeans(matrix: np.ndarray, k: int, iterations: int = 30):
    """Small deterministic NumPy k-means implementation."""
    if len(matrix) == 0:
        return np.empty(0, dtype=int), np.empty((0, matrix.shape[1] if matrix.ndim == 2 else 0))
    k = max(1, min(k, len(matrix)))
    # Deterministic spread across time rather than random initialization.
    indices = np.linspace(0, len(matrix) - 1, k, dtype=int)
    centers = matrix[indices].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        new_centers = centers.copy()
        for cluster in range(k):
            members = matrix[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        centers = new_centers
    return labels, centers


def _cluster_name(center: np.ndarray) -> str:
    direction = center[0]
    alt = center[1]
    rng = center[3]
    if direction > 0.20:
        d = "BULL_BIASED"
    elif direction < -0.20:
        d = "BEAR_BIASED"
    else:
        d = "BALANCED"
    if rng > 0.20:
        r = "WIDE_RANGE"
    elif rng < -0.20:
        r = "TIGHT_RANGE"
    else:
        r = "MID_RANGE"
    if alt > 0.20:
        a = "HIGH_ALTERNATION"
    elif alt < -0.20:
        a = "LOW_ALTERNATION"
    else:
        a = "MID_ALTERNATION"
    return f"{d} | {r} | {a}"


def analyze_timeframe(db: MarketDatabase, timeframe: str, window: int,
                      top_k: int, clusters: int, context: int = 20):
    df = db.load_candles(timeframe)
    if df.empty or len(df) < window * 2:
        return None
    features = candle_features(df)
    current_start = len(features) - window
    current_fp = continuous_fingerprint(features.iloc[current_start:])

    # Sample every second historical window for clustering; all candidates are
    # still used by the similarity ranking. This keeps clustering lightweight.
    fp_matrix, ends = fingerprint_matrix(features, window, step=2)
    valid = ends < current_start
    historical_fp = fp_matrix[valid]
    historical_ends = ends[valid]
    if len(historical_fp) == 0:
        return None

    matrix = standardize_rows(historical_fp)
    current_std = (current_fp - np.nanmedian(historical_fp, axis=0)) / np.where(
        np.nanmedian(np.abs(historical_fp - np.nanmedian(historical_fp, axis=0)), axis=0) * 1.4826 < 1e-6,
        1.0,
        np.nanmedian(np.abs(historical_fp - np.nanmedian(historical_fp, axis=0)), axis=0) * 1.4826,
    )
    labels, centers = kmeans(matrix, clusters)
    current_distances = ((centers - current_std[None, :]) ** 2).mean(axis=1)
    current_cluster = int(current_distances.argmin())

    # Rarity is based on distance to the nearest historical cluster center.
    historical_nearest = ((matrix[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2).min(axis=1)
    current_distance = float(current_distances[current_cluster])
    percentile = float(np.mean(historical_nearest <= current_distance) * 100.0)

    similarity = rolling_similarity(features, window, top_k)
    if not similarity.empty:
        similarity["cluster"] = [
            int(labels[np.searchsorted(historical_ends, end)])
            if end in set(historical_ends) else -1
            for end in similarity["candidate_end"]
        ]
        similarity["cluster_name"] = similarity["cluster"].map(
            {i: _cluster_name(c) for i, c in enumerate(centers)}
        ).fillna("NOT_CLUSTERED")

    counts = pd.Series(labels).value_counts().sort_index()
    cluster_rows = []
    for i, center in enumerate(centers):
        cluster_rows.append({
            "timeframe": timeframe,
            "cluster": i,
            "name": _cluster_name(center),
            "historical_windows": int(counts.get(i, 0)),
            "share_pct": float(counts.get(i, 0) / len(labels) * 100),
            "distance_from_current": float(current_distances[i]),
            "is_current_cluster": bool(i == current_cluster),
        })

    ctx_rows = []
    for end in [*similarity["candidate_end"].tolist()[:top_k]] if not similarity.empty else []:
        start = end - window + 1
        before = features.iloc[max(0, start - context):start]
        matched = features.iloc[start:end + 1]
        after = features.iloc[end + 1:min(len(features), end + 1 + context)]
        ctx_rows.append({
            "timeframe": timeframe,
            "candidate_end": end,
            "before_direction": float(before["direction"].mean()) if len(before) else np.nan,
            "matched_direction": float(matched["direction"].mean()),
            "after_direction": float(after["direction"].mean()) if len(after) else np.nan,
            "before_range": float(before["range_ratio"].mean()) if len(before) else np.nan,
            "matched_range": float(matched["range_ratio"].mean()),
            "after_range": float(after["range_ratio"].mean()) if len(after) else np.nan,
            "before_alternation": float(before["alternation"].mean()) if len(before) else np.nan,
            "matched_alternation": float(matched["alternation"].mean()),
            "after_alternation": float(after["alternation"].mean()) if len(after) else np.nan,
        })

    current_row = {
        "timeframe": timeframe,
        "window": window,
        "current_cluster": current_cluster,
        "current_cluster_name": _cluster_name(centers[current_cluster]),
        "cluster_rarity_percentile": percentile,
        "historical_windows_clustered": len(historical_fp),
        "current_direction_balance": float(current_fp[0]),
        "current_alternation": float(current_fp[1]),
        "current_range_level": float(current_fp[3]),
        "current_body_level": float(current_fp[2]),
        "current_upper_wick": float(current_fp[4]),
        "current_lower_wick": float(current_fp[5]),
        "current_close_location": float(current_fp[6]),
    }
    return current_row, similarity, pd.DataFrame(cluster_rows), pd.DataFrame(ctx_rows)


def run(all_timeframes: bool, timeframe: str, window: int, top_k: int, clusters: int, context: int):
    out_dir = Path("charts")
    out_dir.mkdir(exist_ok=True)
    outputs = [
        "candle_behavior_cluster_current.csv",
        "candle_behavior_cluster_matches.csv",
        "candle_behavior_cluster_summary.csv",
        "candle_behavior_cluster_context.csv",
        "candle_behavior_cluster_rarity.png",
    ]
    for name in outputs:
        path = out_dir / name
        if path.exists():
            path.unlink()

    db = MarketDatabase("data/eth_market.db")
    tfs = TIMEFRAMES if all_timeframes else (timeframe,)
    current_rows, match_rows, cluster_rows, context_rows = [], [], [], []

    for tf in tfs:
        print(f"[{tf}] clustering historical candle behaviour...")
        result = analyze_timeframe(db, tf, window, top_k, clusters, context)
        if result is None:
            print(f"[{tf}] insufficient data")
            continue
        current, matches, summary, ctx = result
        current_rows.append(current)
        if not matches.empty:
            matches.insert(0, "timeframe", tf)
            match_rows.append(matches)
        cluster_rows.append(summary)
        context_rows.append(ctx)
        print(f"[{tf}] cluster={current['current_cluster']} | {current['current_cluster_name']} | rarity percentile={current['cluster_rarity_percentile']:.2f}")

    pd.DataFrame(current_rows).to_csv(out_dir / "candle_behavior_cluster_current.csv", index=False)
    pd.concat(match_rows, ignore_index=True).to_csv(out_dir / "candle_behavior_cluster_matches.csv", index=False)
    pd.concat(cluster_rows, ignore_index=True).to_csv(out_dir / "candle_behavior_cluster_summary.csv", index=False)
    pd.concat(context_rows, ignore_index=True).to_csv(out_dir / "candle_behavior_cluster_context.csv", index=False)

    if current_rows:
        plot = pd.DataFrame(current_rows)
        plt.figure(figsize=(11, 5))
        plt.bar(plot["timeframe"], plot["cluster_rarity_percentile"])
        plt.axhline(50, linewidth=1)
        plt.ylabel("Current cluster-distance percentile")
        plt.xlabel("Timeframe")
        plt.title("Current Candle Behaviour Cluster Rarity")
        plt.tight_layout()
        plt.savefig(out_dir / "candle_behavior_cluster_rarity.png", dpi=150)
        plt.close()

    print("CANDLE BEHAVIOUR CLUSTER RESEARCH")
    print("Continuous descriptive clustering of historical candle behaviour.")
    print(pd.DataFrame(current_rows).to_string(index=False))
    print(f"Saved cluster research outputs in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="1h")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--all-timeframes", action="store_true")
    args = parser.parse_args()
    run(args.all_timeframes, args.timeframe, args.window, args.top_k, args.clusters, args.context)
