"""Fast candle-behaviour clustering and rarity research.

Descriptive research only. This module does not calculate signals, trades,
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
BLOCKS = 5
MACRO_GROUPS = 4
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
DEFAULT_CLUSTERS = 8
SAMPLE_STEP = 10
CHUNK_SIZE = 4096


def rolling_view(values, window):
    if len(values) < window:
        return np.empty((0, window, values.shape[1]), dtype=float)
    return np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)


def standardize_batch(windows):
    med = np.nanmedian(windows, axis=1, keepdims=True)
    scale = np.nanmedian(np.abs(windows - med), axis=1, keepdims=True) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (windows - med) / scale


def similarity_rank(features, window, top_k):
    values = features[list(FEATURES)].to_numpy(float)
    views = rolling_view(values, window)
    current_start = len(values) - window
    candidate_count = current_start - window + 1
    if candidate_count <= 0:
        return pd.DataFrame(columns=["rank", "candidate_start", "candidate_end", "similarity"])
    current = standardize_batch(values[current_start:][None, ...])[0]
    weights = CandleSimilarityConfig().weights
    weights = weights / weights.sum()
    best_scores = np.full(top_k, -np.inf)
    best_ends = np.full(top_k, -1, dtype=int)
    for start in range(0, candidate_count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, candidate_count)
        cand = standardize_batch(views[start:stop])
        per_feature = np.sqrt(np.mean((cand - current[None, ...]) ** 2, axis=1))
        scores = np.exp(-np.sum(per_feature * weights[None, :], axis=1))
        take = min(top_k, len(scores))
        idx = np.argpartition(scores, -take)[-take:]
        cs = np.concatenate([best_scores, scores[idx]])
        ce = np.concatenate([best_ends, np.arange(start, stop)[idx]])
        keep = np.argpartition(cs, -top_k)[-top_k:]
        best_scores, best_ends = cs[keep], ce[keep]
    order = np.argsort(best_scores)[::-1]
    best_scores, best_ends = best_scores[order], best_ends[order]
    valid = best_ends >= 0
    ends = best_ends[valid]
    out = pd.DataFrame({"candidate_start": ends, "candidate_end": ends + window - 1, "similarity": best_scores[valid]})
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def fingerprint_windows(features, window, step):
    """Vectorized continuous fingerprints for sampled rolling windows."""
    values = features[list(FEATURES)].to_numpy(float)
    views = rolling_view(values, window)[::step]
    ends = np.arange(window - 1, len(values), step, dtype=int)
    if not len(views):
        return np.empty((0, 0)), ends
    idx = {name: FEATURES.index(name) for name in FEATURES}
    direction = views[:, :, idx["direction"]]
    body = views[:, :, idx["body_pct"]]
    rng = views[:, :, idx["range_ratio"]]
    upper = views[:, :, idx["upper_wick_pct"]]
    lower = views[:, :, idx["lower_wick_pct"]]
    close = views[:, :, idx["close_location"]]
    alt = views[:, :, idx["alternation"]]
    pieces = [direction.mean(1), alt.mean(1), body.mean(1), rng.mean(1), upper.mean(1), lower.mean(1), close.mean(1), (body <= .25).mean(1), (body >= .60).mean(1), (rng <= .85).mean(1), (rng >= 1.15).mean(1), (close >= .65).mean(1), (close <= .35).mean(1), (direction > 0).mean(1), (direction < 0).mean(1)]
    for groups in (MACRO_GROUPS, BLOCKS):
        if window % groups:
            raise ValueError(f"window must be divisible by {groups}")
        size = window // groups
        for i in range(groups):
            sl = slice(i * size, (i + 1) * size)
            if groups == MACRO_GROUPS:
                pieces.extend([direction[:, sl].mean(1), body[:, sl].mean(1), rng[:, sl].mean(1), close[:, sl].mean(1), alt[:, sl].mean(1)])
            else:
                pieces.extend([direction[:, sl].mean(1), body[:, sl].mean(1), rng[:, sl].mean(1), upper[:, sl].mean(1), lower[:, sl].mean(1), close[:, sl].mean(1), alt[:, sl].mean(1)])
    return np.column_stack(pieces), ends


def robust_standardize(matrix):
    med = np.nanmedian(matrix, axis=0)
    scale = np.nanmedian(np.abs(matrix - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (matrix - med) / scale, med, scale


def kmeans(matrix, k, iterations=25):
    k = max(1, min(k, len(matrix)))
    centers = matrix[np.linspace(0, len(matrix) - 1, k, dtype=int)].copy()
    labels = np.full(len(matrix), -1, dtype=int)
    for _ in range(iterations):
        dist = ((matrix[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            members = matrix[labels == i]
            if len(members):
                centers[i] = members.mean(axis=0)
    return labels, centers


def cluster_name(center):
    d = "BULL_BIASED" if center[0] > .20 else "BEAR_BIASED" if center[0] < -.20 else "BALANCED"
    a = "HIGH_ALTERNATION" if center[1] > .20 else "LOW_ALTERNATION" if center[1] < -.20 else "MID_ALTERNATION"
    r = "WIDE_RANGE" if center[3] > .20 else "TIGHT_RANGE" if center[3] < -.20 else "MID_RANGE"
    return f"{d} | {r} | {a}"


def analyze(db, timeframe, window, top_k, clusters):
    df = db.load_candles("ETHUSDT", timeframe)
    if df.empty or len(df) < window * 2:
        return None
    features = candle_features(df)
    current_start = len(features) - window
    current_fp = fingerprint_windows(features.iloc[current_start:], window, 1)[0][0]
    matrix, ends = fingerprint_windows(features, window, SAMPLE_STEP)
    mask = ends < current_start
    matrix, ends = matrix[mask], ends[mask]
    if len(matrix) < 2:
        return None
    z, med, scale = robust_standardize(matrix)
    current_z = (current_fp - med) / scale
    labels, centers = kmeans(z, clusters)
    center_dist = ((centers - current_z[None, :]) ** 2).mean(axis=1)
    current_cluster = int(center_dist.argmin())
    nearest_hist = ((z[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2).min(axis=1)
    rarity = float(np.mean(nearest_hist <= center_dist[current_cluster]) * 100)

    matches = similarity_rank(features, window, top_k)
    names = {i: cluster_name(c) for i, c in enumerate(centers)}

    # Assign every top match to its nearest cluster, rather than only sampled endpoints.
    match_fps = []
    match_ends = matches["candidate_end"].to_numpy(int) if not matches.empty else np.array([], dtype=int)
    values = features[list(FEATURES)].to_numpy(float)
    for end in match_ends:
        start = end - window + 1
        fp, _ = fingerprint_windows(features.iloc[start:end + 1], window, 1)
        match_fps.append(fp[0])
    if match_fps:
        match_z = (np.vstack(match_fps) - med) / scale
        match_cluster = ((match_z[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2).argmin(axis=1)
        matches["cluster"] = match_cluster
        matches["cluster_name"] = [names[int(i)] for i in match_cluster]
    else:
        matches["cluster"] = pd.Series(dtype=int)
        matches["cluster_name"] = pd.Series(dtype=str)

    counts = np.bincount(labels, minlength=len(centers))
    summary = pd.DataFrame([{"timeframe": timeframe, "cluster": i, "name": names[i], "historical_windows": int(counts[i]), "share_pct": float(counts[i] / len(labels) * 100), "distance_from_current": float(center_dist[i]), "is_current_cluster": bool(i == current_cluster)} for i in range(len(centers))])
    current = {"timeframe": timeframe, "window": window, "current_cluster": current_cluster, "current_cluster_name": names[current_cluster], "cluster_rarity_percentile": rarity, "historical_windows_clustered": len(matrix), "current_direction_balance": float(current_fp[0]), "current_alternation": float(current_fp[1]), "current_body_level": float(current_fp[2]), "current_range_level": float(current_fp[3]), "current_upper_wick": float(current_fp[4]), "current_lower_wick": float(current_fp[5]), "current_close_location": float(current_fp[6])}
    return current, matches, summary


def run(all_timeframes, timeframe, window, top_k, clusters):
    if window % BLOCKS or window % MACRO_GROUPS:
        raise ValueError(f"window must be divisible by both {BLOCKS} and {MACRO_GROUPS}")
    out = Path("charts")
    out.mkdir(exist_ok=True)
    for name in ["candle_behavior_cluster_current.csv", "candle_behavior_cluster_matches.csv", "candle_behavior_cluster_summary.csv", "candle_behavior_cluster_rarity.png"]:
        p = out / name
        if p.exists():
            p.unlink()
    db = MarketDatabase("data/eth_market.db")
    tfs = TIMEFRAMES if all_timeframes else (timeframe,)
    currents, matches_all, summaries = [], [], []
    for tf in tfs:
        print(f"[{tf}] fast candle-behaviour clustering...")
        result = analyze(db, tf, window, top_k, clusters)
        if result is None:
            print(f"[{tf}] insufficient data")
            continue
        current, matches, summary = result
        currents.append(current)
        matches_all.append(matches.assign(timeframe=tf))
        summaries.append(summary)
        print(f"[{tf}] cluster={current['current_cluster']} | {current['current_cluster_name']} | rarity percentile={current['cluster_rarity_percentile']:.2f}")
    current_df = pd.DataFrame(currents)
    matches_df = pd.concat(matches_all, ignore_index=True) if matches_all else pd.DataFrame()
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    current_df.to_csv(out / "candle_behavior_cluster_current.csv", index=False)
    matches_df.to_csv(out / "candle_behavior_cluster_matches.csv", index=False)
    summary_df.to_csv(out / "candle_behavior_cluster_summary.csv", index=False)
    if not current_df.empty:
        plt.figure(figsize=(11, 5))
        plt.bar(current_df["timeframe"], current_df["cluster_rarity_percentile"])
        plt.axhline(50, linewidth=1)
        plt.xlabel("Timeframe")
        plt.ylabel("Cluster-distance percentile")
        plt.title("Current Candle Behaviour Cluster Rarity")
        plt.tight_layout()
        plt.savefig(out / "candle_behavior_cluster_rarity.png", dpi=150)
        plt.close()
    print("CANDLE BEHAVIOUR CLUSTER RESEARCH")
    print(current_df.to_string(index=False))
    print("Saved cluster research outputs in charts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="1h")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--all-timeframes", action="store_true")
    args = parser.parse_args()
    run(args.all_timeframes, args.timeframe, args.window, args.top_k, args.clusters)
