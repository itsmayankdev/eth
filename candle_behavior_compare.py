"""Compare structural matches with raw candle-behaviour similarity.

Research-only tooling. No entries, targets, stops, outcomes, or predictions.

The existing structural matcher supplies historical structural candidates. This
module independently scores the same candle windows using OHLC anatomy and
behaviour, then also finds candle-only historical neighbours. This lets us see
whether candle behaviour agrees with or differs from pivot geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import (
    CandleSimilarityConfig,
    candle_features,
    candle_sequence_similarity,
)
from config import load_config
from data.database import MarketDatabase
from structure.representation import build_structure
from structure.similarity import find_similar_structures


DISPLAY_FEATURES = [
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_ratio",
    "close_location",
    "direction",
    "body_change",
    "range_change",
    "alternation",
]


def load_market(cfg, timeframe: str) -> pd.DataFrame:
    db = MarketDatabase(cfg.database)
    try:
        df = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if df.empty:
        raise ValueError(f"No candle data available for {timeframe}")
    return df.reset_index(drop=True)


def pivot_candle_index(structure: pd.DataFrame, position: int) -> int:
    row = structure.iloc[position]
    return int(row["index"])


def sequence(features: pd.DataFrame, center: int, radius: int) -> pd.DataFrame:
    start = max(0, center - radius)
    end = min(len(features) - 1, center + radius)
    return features.iloc[start : end + 1].copy()


def score_historical_endpoints(
    features: pd.DataFrame,
    current_center: int,
    radius: int,
    candidate_ends: list[int],
    config: CandleSimilarityConfig,
    top_k: int,
) -> pd.DataFrame:
    current = sequence(features, current_center, radius)
    needed = len(current)
    rows = []
    current_start = int(current.index[0])

    for end in candidate_ends:
        start = end - needed + 1
        if start < 0 or end >= current_start:
            continue
        candidate = features.iloc[start : end + 1]
        if len(candidate) != needed:
            continue
        score = candle_sequence_similarity(current, candidate, config)
        rows.append({
            "candidate_start": start,
            "candidate_end": end,
            "similarity": score,
        })

    if not rows:
        return pd.DataFrame(columns=["candidate_start", "candidate_end", "similarity"])
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def plot_candles(ax, raw: pd.DataFrame, title: str, center: int) -> None:
    for i, row in enumerate(raw.itertuples()):
        up = float(row.close) >= float(row.open)
        lo = min(float(row.open), float(row.close))
        height = max(abs(float(row.close) - float(row.open)), 1e-10)
        ax.vlines(i, float(row.low), float(row.high), linewidth=0.8)
        ax.add_patch(
            plt.Rectangle((i - 0.32, lo), 0.64, height, fill=up, linewidth=0.6)
        )
    c = center - int(raw.index[0])
    if 0 <= c < len(raw):
        ax.axvline(c, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xlim(-1, len(raw))
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("USDT")
    ax.grid(alpha=0.12)


def plot_feature_strip(ax, seq_df: pd.DataFrame, title: str) -> None:
    x = np.arange(len(seq_df))
    for feature in ("body_pct", "upper_wick_pct", "lower_wick_pct"):
        ax.plot(x, seq_df[feature].to_numpy() * 100.0, linewidth=1.1, label=feature.replace("_pct", ""))
    ax.axvline(len(seq_df) // 2, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of range")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.12)
    ax.legend(fontsize=7, ncol=3, loc="upper right")


def plot_pairwise(rows: list[dict], output: Path, show: bool) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(16, max(5, 3.8 * n)), squeeze=False)
    for i, item in enumerate(rows):
        cur = item["current"]
        hist = item["historical"]
        base_cur = float(cur["close"].iloc[len(cur) // 2])
        base_hist = float(hist["close"].iloc[len(hist) // 2])
        axes[i, 0].plot(cur["offset"], (cur["close"] / base_cur - 1) * 100, linewidth=1.4)
        axes[i, 1].plot(hist["offset"], (hist["close"] / base_hist - 1) * 100, linewidth=1.4)
        for ax in axes[i]:
            ax.axvline(0, linestyle="--", linewidth=0.7, alpha=0.45)
            ax.axhline(0, linewidth=0.5, alpha=0.25)
            ax.grid(alpha=0.12)
            ax.set_xlabel("Relative candle")
            ax.set_ylabel("Close change from center %")
        axes[i, 0].set_title("CURRENT")
        axes[i, 1].set_title(
            f"{item['title']} | candle similarity {item['candle_similarity']:.4f}"
        )
    fig.suptitle("Current vs historical candle-sequence shape", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def create(
    timeframe: str,
    threshold: float,
    pivots: int,
    top_k: int,
    minimum_similarity: float,
    radius: int,
    candle_top_k: int,
    output_dir: Path,
    show: bool = False,
):
    cfg = load_config()
    raw = load_market(cfg, timeframe)
    features = candle_features(raw)
    structure = build_structure(raw, threshold)
    if len(structure) < pivots * 2:
        raise ValueError("Not enough structural pivots")

    structural = find_similar_structures(
        structure,
        n_pivots=pivots,
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )
    if structural.empty:
        raise ValueError("No structural historical matches")

    current_struct_pos = len(structure) - 1
    current_center = pivot_candle_index(structure, current_struct_pos)
    cfg_sim = CandleSimilarityConfig()
    current_window = sequence(features, current_center, radius)

    candidate_ends = list(range(radius, max(radius, current_center - radius * 2) + 1))
    candle_only = score_historical_endpoints(
        features, current_center, radius, candidate_ends, cfg_sim, candle_top_k
    )

    rows = []
    for rank, (_, match) in enumerate(structural.iterrows(), 1):
        pos = int(match["candidate_end_position"])
        center = pivot_candle_index(structure, pos)
        hist_window = sequence(features, center, radius)
        score = candle_sequence_similarity(current_window, hist_window, cfg_sim)
        rows.append({
            "structural_rank": rank,
            "structural_similarity": float(match["similarity"]),
            "candle_similarity": float(score),
            "candle_end": center,
            "pivot_type_sequence": match.get("pivot_type_sequence", ""),
            "historical": hist_window,
        })

    comparison = pd.DataFrame([
        {k: v for k, v in r.items() if k != "historical"} for r in rows
    ])
    comparison.to_csv(output_dir / "structural_vs_candle_similarity.csv", index=False)

    candle_only_out = candle_only.copy()
    candle_only_out.to_csv(output_dir / "candle_only_top_matches.csv", index=False)

    pair_rows = []
    for r in rows:
        pair_rows.append({
            "title": f"STRUCTURAL #{r['structural_rank']} | structural {r['structural_similarity']:.4f}",
            "candle_similarity": r["candle_similarity"],
            "current": current_window.assign(offset=np.arange(-radius, radius + 1)[:len(current_window)]),
            "historical": r["historical"].assign(offset=np.arange(-radius, radius + 1)[:len(r["historical"])]),
        })
    pair_path = output_dir / "structural_matches_candle_compare.png"
    plot_pairwise(pair_rows, pair_path, show)

    # Dedicated raw-candle comparison for the candle-only neighbours.
    n = len(candle_only)
    fig, axes = plt.subplots(n + 1, 2, figsize=(16, 3.8 * (n + 1)), squeeze=False)
    plot_candles(axes[0, 0], raw.iloc[max(0, current_center - radius):current_center + radius + 1], "CURRENT — raw candles", current_center)
    plot_feature_strip(axes[0, 1], current_window, "CURRENT — candle anatomy")
    for i, (_, m) in enumerate(candle_only.iterrows(), 1):
        center = int(m["candidate_end"])
        z = raw.iloc[max(0, center - radius):center + radius + 1]
        plot_candles(axes[i, 0], z, f"CANDLE-ONLY #{i} — similarity {float(m['similarity']):.4f}", center)
        plot_feature_strip(axes[i, 1], sequence(features, center, radius), "Candle anatomy")
    fig.suptitle(
        f"{cfg.symbol} {timeframe} — Candle-only historical neighbours\n"
        "Raw OHLC and anatomy; descriptive research only.",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    candle_path = output_dir / "candle_only_neighbours.png"
    fig.savefig(candle_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\nSTRUCTURAL MATCHES + CANDLE SIMILARITY")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCANDLE-ONLY MATCHES")
    print(candle_only.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {pair_path}")
    print(f"Saved: {candle_path}")
    print(f"Saved: {output_dir / 'structural_vs_candle_similarity.csv'}")
    print(f"Saved: {output_dir / 'candle_only_top_matches.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare structural and candle-behaviour similarity.")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--pivots", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--minimum-similarity", type=float, default=0.70)
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--candle-top-k", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("charts"))
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    create(**vars(a))


if __name__ == "__main__":
    main()
