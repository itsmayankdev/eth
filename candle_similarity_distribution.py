"""Descriptive distribution analysis for historical candle similarity."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from candle_similarity import FEATURES, CandleSimilarityConfig, candle_features, candle_sequence_similarity
from config import load_config
from data.database import MarketDatabase


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        df = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if df.empty:
        raise ValueError(f"No candle data available for {timeframe}")
    return df.reset_index(drop=True)


def sequence(features, end, window):
    start = end - window + 1
    if start < 0 or end >= len(features):
        raise IndexError("Invalid sequence bounds")
    return features.iloc[start:end + 1]


def rank_all(features, current_end, window, cfg):
    current_start = current_end - window + 1
    current = sequence(features, current_end, window)
    rows = []
    for end in range(window - 1, current_start):
        candidate = sequence(features, end, window)
        score = candle_sequence_similarity(current, candidate, cfg)
        rows.append({"candidate_start": end - window + 1, "candidate_end": end, "similarity": score})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def feature_differences(current, candidate, cfg):
    a = current[list(FEATURES)].to_numpy(float)
    b = candidate[list(FEATURES)].to_numpy(float)

    def z(x):
        med = np.nanmedian(x, axis=0)
        scale = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
        return (x - med) / np.where(scale < 1e-6, 1.0, scale)

    rmse = np.sqrt(np.mean((z(a) - z(b)) ** 2, axis=0))
    w = cfg.weights / cfg.weights.sum()
    return (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "rmse": rmse,
                "weight": w,
                "weighted_difference": rmse * w,
            }
        )
        .sort_values("weighted_difference", ascending=False)
        .reset_index(drop=True)
    )


def draw_candles(ax, df, title):
    for i, r in enumerate(df.itertuples()):
        lo = min(float(r.open), float(r.close))
        h = max(abs(float(r.close) - float(r.open)), 1e-10)
        ax.vlines(i, float(r.low), float(r.high), linewidth=0.7)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.32, lo),
                0.64,
                h,
                fill=float(r.close) >= float(r.open),
                linewidth=0.5,
            )
        )
    ax.set_xlim(-1, len(df))
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.12)


def create(timeframe, window, top_k, output_dir, show=False):
    cfg = load_config()
    raw = load_market(cfg, timeframe)
    features = candle_features(raw)
    sim_cfg = CandleSimilarityConfig()
    current_end = len(raw) - 1

    matches = rank_all(features, current_end, window, sim_cfg)
    if matches.empty:
        raise ValueError("No historical candle windows available")

    top = matches.head(top_k).copy()
    scores = matches["similarity"].to_numpy(float)
    best = float(top.iloc[0].similarity)
    current = sequence(features, current_end, window)
    best_candidate = sequence(features, int(top.iloc[0].candidate_end), window)
    contrib = feature_differences(current, best_candidate, sim_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_dir / "candle_similarity_distribution_matches.csv", index=False)
    contrib.to_csv(output_dir / "candle_similarity_feature_contributions.csv", index=False)

    # The CLI --show flag used to import matplotlib.pyplot inside this function.
    # That made pyplot a local variable and caused UnboundLocalError earlier.
    # pyplot is now imported only at module scope, so every plotting section can
    # safely use plt and --show simply displays the already-created figures.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores, bins=60, density=True, alpha=0.75)
    ax.axvline(best, linestyle="--", linewidth=1.3, label=f"Best = {best:.4f}")
    ax.set_title("Historical candle-sequence similarity distribution")
    ax.set_xlabel("Candle similarity")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "candle_similarity_distribution.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    n = len(top)
    fig, axes = plt.subplots(
        n + 1,
        1,
        figsize=(14, max(6, 2.7 * (n + 1))),
        squeeze=False,
    )
    axes = axes[:, 0]
    draw_candles(
        axes[0],
        raw.iloc[current_end - window + 1 : current_end + 1],
        f"CURRENT — last {window} candles",
    )
    for i, r in top.iterrows():
        hist = raw.iloc[int(r.candidate_start) : int(r.candidate_end) + 1]
        draw_candles(
            axes[i + 1],
            hist,
            f"#{i+1} — similarity {r.similarity:.4f} — {int(r.candidate_start)}..{int(r.candidate_end)}",
        )
    fig.suptitle(
        f"{cfg.symbol} {timeframe} — strongest historical candle neighbours",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "candle_similarity_top_sequences.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(contrib))
    ax.bar(x, contrib.weighted_difference)
    ax.set_xticks(x)
    ax.set_xticklabels(contrib.feature, rotation=35, ha="right")
    ax.set_ylabel("Weighted normalized difference")
    ax.set_title("Candle features driving difference to best match")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "candle_similarity_feature_contributions.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    print("\nCANDLE SIMILARITY DISTRIBUTION")
    print(f"Symbol: {cfg.symbol} | timeframe: {timeframe} | window: {window} candles")
    print(f"Historical non-overlapping windows: {len(scores)}")
    print(f"Best similarity: {best:.4f} | percentile: {np.mean(scores <= best) * 100:.2f}")
    print(
        f"Median: {np.median(scores):.4f} | P90: {np.percentile(scores, 90):.4f} | "
        f"P95: {np.percentile(scores, 95):.4f} | P99: {np.percentile(scores, 99):.4f}"
    )
    print("\nTOP HISTORICAL CANDLE SEQUENCES")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFEATURE DIFFERENCES VS BEST MATCH")
    print(contrib.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved charts/csv outputs in", output_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("charts"))
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    create(**vars(a))


if __name__ == "__main__":
    main()
