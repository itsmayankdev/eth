from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from config import load_config
from data.database import MarketDatabase
from structure.representation import build_structure
from structure.similarity import find_similar_structures


def _candle_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.6
    return max(0.15, float(np.median(np.diff(x))) * 0.65)


def _plot_candles(ax, candles: pd.DataFrame, start: int, end: int) -> None:
    frame = candles.iloc[max(0, start):min(len(candles), end + 1)].copy().reset_index(drop=True)
    if frame.empty:
        return
    x = np.arange(start, start + len(frame), dtype=float)
    width = _candle_width(x)
    for i, row in frame.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        up = c >= o
        body_low = min(o, c)
        body_height = max(abs(c - o), 1e-9)
        ax.vlines(x[i], l, h, linewidth=0.8)
        rect = Rectangle(
            (x[i] - width / 2, body_low), width, body_height,
            fill=up, linewidth=0.8,
        )
        ax.add_patch(rect)


def _annotate_structure(
    ax,
    structure: pd.DataFrame,
    candles: pd.DataFrame,
    start_pos: int,
    end_pos: int,
    label_prefix: str,
) -> None:
    pivots = structure.iloc[start_pos:end_pos + 1].copy()
    if pivots.empty:
        return
    labels = []
    for _, pivot in pivots.iterrows():
        x = int(pivot["index"])
        y = float(pivot["price"])
        label = str(pivot["pivot_type"])
        labels.append(label)
        ax.scatter([x], [y], s=35, zorder=5)
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 10 if str(pivot["direction"]) == "UP" else -16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    if len(pivots) >= 2:
        ax.plot(
            pivots["index"].astype(float).to_numpy(),
            pivots["price"].astype(float).to_numpy(),
            linewidth=1.2,
        )
    first = int(pivots.iloc[0]["index"])
    last = int(pivots.iloc[-1]["index"])
    ax.axvline(first, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(last, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{label_prefix}: pivots {' → '.join(labels)}")


def _plot_confirmation(ax, structure_row: pd.Series, candles: pd.DataFrame, label: str) -> None:
    confirmation = int(structure_row["confirmation_index"])
    if confirmation < 0 or confirmation >= len(candles):
        return
    price = float(candles.iloc[confirmation]["close"])
    ax.axvline(confirmation, linestyle=":", linewidth=1.2)
    ax.scatter([confirmation], [price], s=45, zorder=6)
    ax.annotate(
        f"{label}\nconfirmation {confirmation}\nentry {price:.2f}",
        (confirmation, price),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
    )


def _plot_outcome_zone(ax, candles: pd.DataFrame, entry_index: int, entry_price: float, target_pct: float, stop_pct: float, horizon: int) -> None:
    target = entry_price * (1.0 + target_pct / 100.0)
    stop = entry_price * (1.0 - stop_pct / 100.0)
    right = min(len(candles) - 1, entry_index + horizon)
    ax.axhline(target, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(stop, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(right, target, f"  +{target_pct:.2f}% target", va="bottom", fontsize=8)
    ax.text(right, stop, f"  -{stop_pct:.2f}% stop", va="top", fontsize=8)
    ax.axvspan(entry_index, right, alpha=0.05)


def create_chart(
    timeframe: str,
    threshold: float,
    pivots: int,
    top_k: int,
    minimum_similarity: float,
    context_candles: int,
    outcome_horizon: int,
    output: Path,
    show: bool = False,
) -> Path:
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    try:
        candles = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()

    if candles.empty:
        raise ValueError(f"No candle data available for {timeframe}")

    structure = build_structure(candles, threshold)
    if len(structure) < pivots * 2:
        raise ValueError(f"Not enough confirmed pivots: {len(structure)}")

    matches = find_similar_structures(
        structure,
        n_pivots=pivots,
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )
    if matches.empty:
        raise ValueError("No historical matches met the requested similarity threshold")

    current_start_pos = len(structure) - pivots
    current_end_pos = len(structure) - 1
    current_start_index = int(structure.iloc[current_start_pos]["index"])
    current_end_index = int(structure.iloc[current_end_pos]["index"])
    current_confirmation = int(structure.iloc[current_end_pos]["confirmation_index"])

    n_panels = 1 + len(matches)
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 5.0 * n_panels), squeeze=False)
    axes = axes[:, 0]

    current_left = max(0, current_start_index - context_candles)
    current_right = min(len(candles) - 1, current_confirmation + outcome_horizon)
    _plot_candles(axes[0], candles, current_left, current_right)
    _annotate_structure(axes[0], structure, candles, current_start_pos, current_end_pos, "CURRENT")
    _plot_confirmation(axes[0], structure.iloc[current_end_pos], candles, "CURRENT")
    axes[0].set_ylabel("USDT")
    axes[0].grid(alpha=0.15)

    for panel, (_, match) in enumerate(matches.iterrows(), start=1):
        candidate_end_pos = int(match["candidate_end_position"])
        candidate_start_pos = candidate_end_pos - pivots + 1
        candidate_start_index = int(structure.iloc[candidate_start_pos]["index"])
        candidate_confirmation = int(match["candidate_confirmation_index"])
        entry_price = float(candles.iloc[candidate_confirmation]["close"])
        left = max(0, candidate_start_index - context_candles)
        right = min(len(candles) - 1, candidate_confirmation + outcome_horizon)
        ax = axes[panel]
        _plot_candles(ax, candles, left, right)
        _annotate_structure(ax, structure, candles, candidate_start_pos, candidate_end_pos, f"MATCH #{panel}: similarity {float(match['similarity']):.4f}")
        _plot_confirmation(ax, structure.iloc[candidate_end_pos], candles, "MATCH")
        _plot_outcome_zone(ax, candles, candidate_confirmation, entry_price, cfg.outcomes.target_pct, cfg.outcomes.stop_pct, outcome_horizon)
        ax.set_ylabel("USDT")
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"{cfg.symbol} {timeframe} — Historical Structure Matching\n"
        f"ZigZag {threshold:.2f}% | {pivots} pivots | min similarity {minimum_similarity:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Candle index (database row)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize current and historical ETH market-structure matches.")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--pivots", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-similarity", type=float, default=0.70)
    parser.add_argument("--context-candles", type=int, default=25)
    parser.add_argument("--outcome-horizon", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("charts/structure_matches.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path = create_chart(
        args.timeframe,
        args.threshold,
        args.pivots,
        args.top_k,
        args.minimum_similarity,
        args.context_candles,
        args.outcome_horizon,
        args.output,
        args.show,
    )
    print(f"Chart saved: {path}")


if __name__ == "__main__":
    main()
