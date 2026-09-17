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


PIVOT_TYPES = {"HH", "HL", "LH", "LL"}


def _candle_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.6
    return max(0.15, float(np.median(np.diff(x))) * 0.65)


def _candle_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive OHLC candle-anatomy features."""
    out = candles.copy()
    o = pd.to_numeric(out["open"], errors="coerce")
    h = pd.to_numeric(out["high"], errors="coerce")
    l = pd.to_numeric(out["low"], errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")

    rng = (h - l).clip(lower=1e-12)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    out["range_pct"] = rng / c.replace(0, np.nan) * 100.0
    out["body_pct_of_range"] = body / rng * 100.0
    out["upper_wick_pct"] = upper / rng * 100.0
    out["lower_wick_pct"] = lower / rng * 100.0
    out["close_location_pct"] = (c - l) / rng * 100.0
    out["direction"] = np.where(c >= o, "UP", "DOWN")
    median_range = out["range_pct"].rolling(20, min_periods=5).median()
    out["range_vs_median"] = out["range_pct"] / median_range.replace(0, np.nan)
    return out


def _regime_features(candles: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Create observational compression/expansion descriptors.

    These are descriptive measurements only. They are not trading signals or
    predictions.
    """
    out = candles.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    changes = close.diff().abs()

    rolling_high = high.rolling(window, min_periods=max(3, window // 2)).max()
    rolling_low = low.rolling(window, min_periods=max(3, window // 2)).min()
    range_pct = (rolling_high - rolling_low) / close.replace(0, np.nan) * 100.0
    path = changes.rolling(window, min_periods=max(3, window // 2)).sum()
    net = close.diff(window).abs()
    efficiency = net / path.replace(0, np.nan)

    range_baseline = range_pct.rolling(window * 2, min_periods=max(6, window)).median()
    out["rolling_range_pct"] = range_pct
    out["efficiency"] = efficiency
    out["compression_ratio"] = range_pct / range_baseline.replace(0, np.nan)
    return out


def _plot_candles(ax, candles: pd.DataFrame, start: int, end: int) -> None:
    frame = candles.iloc[max(0, start):min(len(candles), end + 1)].copy().reset_index(drop=True)
    if frame.empty:
        return
    x = np.arange(start, start + len(frame), dtype=float)
    width = _candle_width(x)
    for i, row in frame.iterrows():
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        up = c >= o
        body_low = min(o, c)
        body_height = max(abs(c - o), 1e-9)
        ax.vlines(x[i], l, h, linewidth=0.8)
        rect = Rectangle((x[i] - width / 2, body_low), width, body_height, fill=up, linewidth=0.8)
        ax.add_patch(rect)


def _behaviour_label(row: pd.Series) -> str:
    compression = row.get("compression_ratio", np.nan)
    efficiency = row.get("efficiency", np.nan)
    range_vs_median = row.get("range_vs_median", np.nan)
    if pd.notna(compression) and pd.notna(efficiency):
        if compression < 0.75 and efficiency < 0.35:
            return "compression / sideways-like"
        if pd.notna(range_vs_median) and range_vs_median > 1.5 and efficiency > 0.55:
            return "expansion / directional-like"
    return "mixed / transitional"


def _pivot_anatomy_text(candles: pd.DataFrame, pivot_index: int) -> str:
    if pivot_index < 0 or pivot_index >= len(candles):
        return ""
    r = candles.iloc[pivot_index]
    return (
        f"body {r['body_pct_of_range']:.0f}% | "
        f"upper {r['upper_wick_pct']:.0f}% | "
        f"lower {r['lower_wick_pct']:.0f}% | "
        f"range×median {r['range_vs_median']:.1f}"
    )


def _annotate(ax, structure: pd.DataFrame, candles: pd.DataFrame, start_pos: int, end_pos: int, title: str) -> None:
    pivots = structure.iloc[start_pos:end_pos + 1].copy()
    if pivots.empty:
        return

    labels: list[str] = []
    for _, pivot in pivots.iterrows():
        x = int(pivot["index"])
        y = float(pivot["price"])
        label = str(pivot["pivot_type"])
        labels.append(label)
        ax.scatter([x], [y], s=38, zorder=5)
        offset_y = 12 if str(pivot["direction"]) == "UP" else -18
        ax.annotate(label, (x, y), xytext=(0, offset_y), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        anatomy = _pivot_anatomy_text(candles, x)
        if anatomy:
            ax.annotate(anatomy, (x, y), xytext=(0, 25 if offset_y > 0 else -38), textcoords="offset points", ha="center", fontsize=6.5, alpha=0.8)

    if len(pivots) >= 2:
        ax.plot(pivots["index"].astype(float), pivots["price"].astype(float), linewidth=1.1, alpha=0.7)
    first = int(pivots.iloc[0]["index"])
    last = int(pivots.iloc[-1]["index"])
    ax.axvline(first, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axvline(last, linestyle="--", linewidth=0.8, alpha=0.4)

    last_row = candles.iloc[last]
    behaviour = _behaviour_label(last_row)
    ax.set_title(f"{title}\nStructure: {' → '.join(labels)} | endpoint context: {behaviour}", fontsize=10)


def _add_context_band(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    """Mark broad descriptive regimes using transparent observational metrics."""
    if right <= left:
        return
    frame = candles.iloc[left:right + 1]
    for idx in range(left, right + 1):
        row = candles.iloc[idx]
        label = _behaviour_label(row)
        if label == "compression / sideways-like":
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.025)
        elif label == "expansion / directional-like":
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.045)


def _plot_feature_strip(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    x = np.arange(left, right + 1)
    frame = candles.iloc[left:right + 1]
    ax.plot(x, frame["body_pct_of_range"].to_numpy(), linewidth=1.0, label="body % of range")
    ax.plot(x, frame["upper_wick_pct"].to_numpy(), linewidth=0.8, label="upper wick %")
    ax.plot(x, frame["lower_wick_pct"].to_numpy(), linewidth=0.8, label="lower wick %")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Candle anatomy (%)")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=7, ncol=3)


def _prepare(candles: pd.DataFrame) -> pd.DataFrame:
    out = _candle_features(candles)
    out = _regime_features(out)
    return out


def create_chart(
    timeframe: str,
    threshold: float,
    pivots: int,
    top_k: int,
    minimum_similarity: float,
    context_candles: int,
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

    candles = _prepare(candles)
    structure = build_structure(candles, threshold)
    if len(structure) < pivots * 2:
        raise ValueError(f"Not enough confirmed pivots: {len(structure)}")

    matches = find_similar_structures(structure, n_pivots=pivots, top_k=top_k, minimum_similarity=minimum_similarity)
    if matches.empty:
        raise ValueError("No historical matches met the requested similarity threshold")

    current_start_pos = len(structure) - pivots
    current_end_pos = len(structure) - 1
    current_start_index = int(structure.iloc[current_start_pos]["index"])
    current_end_index = int(structure.iloc[current_end_pos]["index"])
    current_left = max(0, current_start_index - context_candles)
    current_right = min(len(candles) - 1, current_end_index + max(3, context_candles // 3))

    n_panels = 1 + len(matches)
    fig, axes = plt.subplots(n_panels, 2, figsize=(20, 4.8 * n_panels), gridspec_kw={"width_ratios": [4, 1.6]}, squeeze=False)

    _plot_candles(axes[0, 0], candles, current_left, current_right)
    _add_context_band(axes[0, 0], candles, current_left, current_right)
    _annotate(axes[0, 0], structure, candles, current_start_pos, current_end_pos, "CURRENT — actual OHLC candles")
    _plot_feature_strip(axes[0, 1], candles, current_left, current_right)
    axes[0, 0].set_ylabel("USDT")

    for panel, (_, match) in enumerate(matches.iterrows(), start=1):
        end_pos = int(match["candidate_end_position"])
        start_pos = end_pos - pivots + 1
        start_index = int(structure.iloc[start_pos]["index"])
        end_index = int(structure.iloc[end_pos]["index"])
        left = max(0, start_index - context_candles)
        right = min(len(candles) - 1, end_index + max(3, context_candles // 3))
        ax = axes[panel, 0]
        _plot_candles(ax, candles, left, right)
        _add_context_band(ax, candles, left, right)
        _annotate(ax, structure, candles, start_pos, end_pos, f"HISTORICAL MATCH #{panel} — similarity {float(match['similarity']):.4f}")
        _plot_feature_strip(axes[panel, 1], candles, left, right)
        ax.set_ylabel("USDT")

    fig.suptitle(
        f"{cfg.symbol} {timeframe} — Candle Behaviour Study\n"
        f"Actual OHLC anatomy + structural context | ZigZag {threshold:.2f}% | {pivots} pivots | min similarity {minimum_similarity:.2f}\n"
        "Compression/expansion markings are descriptive measurements, not signals or predictions.",
        fontsize=14,
        fontweight="bold",
    )
    axes[-1, 0].set_xlabel("Candle index (database row)")
    axes[-1, 1].set_xlabel("Candle index")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Study actual ETH candle behaviour around structural patterns.")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--pivots", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-similarity", type=float, default=0.70)
    parser.add_argument("--context-candles", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("charts/candle_behavior.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path = create_chart(
        args.timeframe,
        args.threshold,
        args.pivots,
        args.top_k,
        args.minimum_similarity,
        args.context_candles,
        args.output,
        args.show,
    )
    print(f"Chart saved: {path}")


if __name__ == "__main__":
    main()
