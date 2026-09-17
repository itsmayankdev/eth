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


STATE_NAMES = {
    0: "compression",
    1: "balanced",
    2: "expansion",
    3: "directional",
    4: "rejection",
}


def _candle_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.6
    return max(0.15, float(np.median(np.diff(x))) * 0.65)


def _prepare(candles: pd.DataFrame, window: int = 12) -> pd.DataFrame:
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
    out["body_pct"] = body / rng * 100.0
    out["upper_wick_pct"] = upper / rng * 100.0
    out["lower_wick_pct"] = lower / rng * 100.0
    out["close_location_pct"] = (c - l) / rng * 100.0
    out["candle_direction"] = np.where(c >= o, "UP", "DOWN")

    median_range = out["range_pct"].rolling(20, min_periods=5).median()
    out["range_vs_median"] = out["range_pct"] / median_range.replace(0, np.nan)

    rolling_high = h.rolling(window, min_periods=max(3, window // 2)).max()
    rolling_low = l.rolling(window, min_periods=max(3, window // 2)).min()
    rolling_range = (rolling_high - rolling_low) / c.replace(0, np.nan) * 100.0
    path = c.diff().abs().rolling(window, min_periods=max(3, window // 2)).sum()
    net = c.diff(window).abs()
    efficiency = net / path.replace(0, np.nan)
    baseline = rolling_range.rolling(window * 2, min_periods=max(6, window)).median()

    out["rolling_range_pct"] = rolling_range
    out["efficiency"] = efficiency
    out["compression_ratio"] = rolling_range / baseline.replace(0, np.nan)

    # Descriptive state only. It does not represent a trading signal.
    state = np.full(len(out), 1, dtype=int)
    compression = (out["compression_ratio"] < 0.75) & (out["efficiency"] < 0.35)
    expansion = (out["range_vs_median"] > 1.5)
    directional = expansion & (out["efficiency"] > 0.55)
    rejection = (
        ((out["upper_wick_pct"] > 50) | (out["lower_wick_pct"] > 50))
        & (out["body_pct"] < 40)
        & (out["range_vs_median"] > 1.15)
    )
    state[compression.fillna(False).to_numpy()] = 0
    state[expansion.fillna(False).to_numpy()] = 2
    state[directional.fillna(False).to_numpy()] = 3
    state[rejection.fillna(False).to_numpy()] = 4
    out["behaviour_state"] = state
    out["behaviour_name"] = pd.Series(state, index=out.index).map(STATE_NAMES)
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
        rect = Rectangle(
            (x[i] - width / 2, body_low),
            width,
            body_height,
            fill=up,
            linewidth=0.8,
        )
        ax.add_patch(rect)


def _plot_behaviour_strip(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    """Show a compact categorical behaviour strip instead of noisy feature lines."""
    frame = candles.iloc[left:right + 1]
    states = frame["behaviour_state"].to_numpy(dtype=int)
    x = np.arange(left, right + 1)
    ax.step(x, states, where="mid", linewidth=1.5)
    ax.set_yticks(list(STATE_NAMES))
    ax.set_yticklabels([STATE_NAMES[i] for i in STATE_NAMES], fontsize=7)
    ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel("Behaviour")
    ax.grid(axis="x", alpha=0.12)


def _plot_metric_strip(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    """Show the two most useful continuous descriptors: range and efficiency."""
    frame = candles.iloc[left:right + 1]
    x = np.arange(left, right + 1)
    range_ratio = frame["range_vs_median"].clip(0, 3).to_numpy(dtype=float)
    efficiency = frame["efficiency"].clip(0, 1).to_numpy(dtype=float)
    ax.plot(x, range_ratio, linewidth=1.0, label="range / median")
    ax.plot(x, efficiency, linewidth=1.0, label="directional efficiency")
    ax.axhline(1.0, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylim(0, 3)
    ax.set_ylabel("Relative behaviour")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.12)


def _add_state_bands(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    """Lightly mark contiguous descriptive behaviour regions on the candle chart."""
    frame = candles.iloc[left:right + 1]
    states = frame["behaviour_state"].to_numpy(dtype=int)
    if len(states) == 0:
        return
    start = left
    current = int(states[0])
    for offset in range(1, len(states) + 1):
        changed = offset == len(states) or int(states[offset]) != current
        if not changed:
            continue
        end = left + offset - 1
        alpha = {0: 0.035, 1: 0.0, 2: 0.025, 3: 0.02, 4: 0.03}.get(current, 0.0)
        if alpha:
            ax.axvspan(start - 0.5, end + 0.5, alpha=alpha)
        if offset < len(states):
            start = left + offset
            current = int(states[offset])


def _pivot_rows(structure: pd.DataFrame, start_pos: int, end_pos: int, candles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos in range(start_pos, end_pos + 1):
        p = structure.iloc[pos]
        idx = int(p["index"])
        c = candles.iloc[idx]
        rows.append(
            {
                "pivot": str(p["pivot_type"]),
                "index": idx,
                "body": float(c["body_pct"]),
                "upper": float(c["upper_wick_pct"]),
                "lower": float(c["lower_wick_pct"]),
                "range_x": float(c["range_vs_median"]),
                "state": str(c["behaviour_name"]),
            }
        )
    return pd.DataFrame(rows)


def _add_pivot_markers(ax, structure: pd.DataFrame, candles: pd.DataFrame, start_pos: int, end_pos: int) -> None:
    pivots = structure.iloc[start_pos:end_pos + 1]
    for _, p in pivots.iterrows():
        idx = int(p["index"])
        price = float(p["price"])
        ax.scatter([idx], [price], s=30, zorder=6)
        offset = 10 if str(p["direction"]) == "UP" else -14
        ax.annotate(
            str(p["pivot_type"]),
            (idx, price),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )


def _add_transition_markers(ax, candles: pd.DataFrame, left: int, right: int) -> None:
    states = candles["behaviour_state"].to_numpy(dtype=int)
    for idx in range(max(left + 1, 1), min(right + 1, len(candles))):
        if states[idx] != states[idx - 1]:
            ax.axvline(idx, linestyle=":", linewidth=0.65, alpha=0.35)


def _summary_text(candles: pd.DataFrame, left: int, right: int) -> str:
    frame = candles.iloc[left:right + 1]
    counts = frame["behaviour_name"].value_counts()
    parts = [f"{name}: {int(counts.get(name, 0))}" for name in STATE_NAMES.values() if counts.get(name, 0)]
    return " | ".join(parts) if parts else "No behaviour measurements"


def _render_panel(
    axes: np.ndarray,
    candles: pd.DataFrame,
    structure: pd.DataFrame,
    start_pos: int,
    end_pos: int,
    left: int,
    right: int,
    title: str,
) -> None:
    price_ax, state_ax, metric_ax = axes
    _plot_candles(price_ax, candles, left, right)
    _add_state_bands(price_ax, candles, left, right)
    _add_transition_markers(price_ax, candles, left, right)
    _add_pivot_markers(price_ax, structure, candles, start_pos, end_pos)
    price_ax.set_title(f"{title}\n{_summary_text(candles, left, right)}", fontsize=10)
    price_ax.set_ylabel("USDT")
    price_ax.grid(alpha=0.12)

    _plot_behaviour_strip(state_ax, candles, left, right)
    _plot_metric_strip(metric_ax, candles, left, right)
    metric_ax.set_xlabel("Candle index")


def _print_pivot_table(name: str, structure: pd.DataFrame, candles: pd.DataFrame, start_pos: int, end_pos: int) -> None:
    table = _pivot_rows(structure, start_pos, end_pos, candles)
    print(f"\n{name}")
    print(table.to_string(index=False, formatters={
        "body": "{:.0f}%".format,
        "upper": "{:.0f}%".format,
        "lower": "{:.0f}%".format,
        "range_x": "{:.1f}x".format,
    }))


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
    current_left = max(0, current_start_index - context_candles)
    current_right = min(len(candles) - 1, current_end_index + max(3, context_candles // 3))

    rows = 1 + len(matches)
    fig, axes = plt.subplots(
        rows * 3,
        1,
        figsize=(19, 4.6 * rows),
        gridspec_kw={"height_ratios": sum(([3.4, 1.0, 1.2] for _ in range(rows)), [])},
        squeeze=False,
    )
    flat = axes[:, 0]

    panel_axes = flat[0:3]
    _render_panel(
        panel_axes,
        candles,
        structure,
        current_start_pos,
        current_end_pos,
        current_left,
        current_right,
        "CURRENT — actual candle behaviour",
    )
    _print_pivot_table("CURRENT pivot candle anatomy", structure, candles, current_start_pos, current_end_pos)

    for panel, (_, match) in enumerate(matches.iterrows(), start=1):
        end_pos = int(match["candidate_end_position"])
        start_pos = end_pos - pivots + 1
        start_index = int(structure.iloc[start_pos]["index"])
        end_index = int(structure.iloc[end_pos]["index"])
        left = max(0, start_index - context_candles)
        right = min(len(candles) - 1, end_index + max(3, context_candles // 3))
        panel_axes = flat[panel * 3:(panel + 1) * 3]
        _render_panel(
            panel_axes,
            candles,
            structure,
            start_pos,
            end_pos,
            left,
            right,
            f"HISTORICAL MATCH #{panel} — structural similarity {float(match['similarity']):.4f}",
        )
        _print_pivot_table(f"MATCH #{panel} pivot candle anatomy", structure, candles, start_pos, end_pos)

    fig.suptitle(
        f"{cfg.symbol} {timeframe} — Candle Behaviour Research\n"
        f"Actual OHLC candles + behaviour states + structural context | ZigZag {threshold:.2f}% | {pivots} pivots\n"
        "Behaviour states are descriptive measurements, not trading signals, outcomes, or predictions.",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Research actual ETH candle behaviour around structural patterns.")
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
