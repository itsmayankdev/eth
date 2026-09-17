"""Price-level interaction research engine.

Describes how a candle sequence interacts with earlier price areas. This is
research-only: no entries, targets, stops, outcomes, or predictions.

The engine focuses on the next layer beyond candle shape:
- prior candle highs/lows/body zones
- local swing highs/lows
- revisits and repeated tests
- breaks and holds
- rejection from a prior area
- pullback depth
- reaction after an interaction
- renewed range/body expansion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import candle_features, rank_candle_sequences
from config import load_config
from data.database import MarketDatabase


DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
EPS = 1e-12


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        raw = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if raw.empty:
        raise ValueError(f"No candle data available for {timeframe}")
    raw = raw.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return raw


def _zone_bounds(price: float, reference_range: float, tolerance: float) -> tuple[float, float]:
    width = max(reference_range * tolerance, EPS)
    return price - width, price + width


def _interaction_for_window(raw: pd.DataFrame, features: pd.DataFrame, start: int, end: int) -> dict:
    """Create a descriptive level-interaction story for one candle window."""
    w = raw.iloc[start:end + 1].reset_index(drop=True)
    f = features.iloc[start:end + 1].reset_index(drop=True)
    if len(w) < 3:
        return {"interaction_count": 0, "story": "INSUFFICIENT_WINDOW"}

    o = w.open.to_numpy(float)
    h = w.high.to_numpy(float)
    l = w.low.to_numpy(float)
    c = w.close.to_numpy(float)
    rng = np.maximum(h - l, EPS)
    body = np.abs(c - o)
    direction = np.sign(c - o)

    # A level is represented by an earlier candle's high, low, and body zone.
    # The tolerance scales with the earlier candle's own range, preventing a
    # fixed-price threshold from behaving differently across volatility regimes.
    events = []
    first_revisit = None
    last_interaction = None
    repeated_level_hits = 0
    break_count = 0
    hold_count = 0
    rejection_count = 0
    pullback_count = 0
    reaction_expansion_count = 0
    interaction_prices = []
    interaction_types = []

    # Compare each candle with the immediately preceding local history inside
    # the window. Keep the candidate level with the strongest/closest contact.
    for i in range(1, len(w)):
        prior_hi = h[:i]
        prior_lo = l[:i]
        prior_body_hi = np.maximum(o[:i], c[:i])
        prior_body_lo = np.minimum(o[:i], c[:i])
        prior_range = rng[:i]
        current_range = rng[i]
        tolerance = 0.20

        best = None
        for j in range(i):
            candidates = [
                ("PRIOR_HIGH", prior_hi[j], "high"),
                ("PRIOR_LOW", prior_lo[j], "low"),
            ]
            body_mid = (prior_body_hi[j] + prior_body_lo[j]) / 2.0
            candidates.append(("PRIOR_BODY", body_mid, "body"))
            for label, level, kind in candidates:
                distance = min(abs(h[i] - level), abs(l[i] - level), abs(c[i] - level))
                threshold = max(prior_range[j] * tolerance, EPS)
                normalized = distance / threshold
                if normalized <= 1.0 and (best is None or normalized < best[0]):
                    best = (normalized, label, level, j, kind)

        if best is None:
            continue

        _, label, level, source_idx, kind = best
        touched = (l[i] <= level <= h[i]) or abs(c[i] - level) <= max(prior_range[source_idx] * tolerance, EPS)
        if not touched:
            continue

        interaction_count = 1
        repeated_level_hits += int(sum(abs(level - p) <= max(rng[source_idx] * tolerance, EPS) for p in interaction_prices) > 0)
        interaction_prices.append(float(level))
        interaction_types.append(label)
        last_interaction = i
        if first_revisit is None:
            first_revisit = i

        # Break: candle closes beyond the prior level in the direction of travel.
        if label == "PRIOR_HIGH" and c[i] > level:
            break_count += 1
            event_type = "BREAK_ABOVE_PRIOR_HIGH"
        elif label == "PRIOR_LOW" and c[i] < level:
            break_count += 1
            event_type = "BREAK_BELOW_PRIOR_LOW"
        else:
            event_type = "LEVEL_REVISIT"

        # Hold: price touches the area but closes back on the same side as the
        # earlier level's surrounding side rather than closing through it.
        if label == "PRIOR_HIGH" and c[i] <= level:
            hold_count += 1
        elif label == "PRIOR_LOW" and c[i] >= level:
            hold_count += 1

        # Rejection: large wick through/into the area with a relatively smaller body.
        body_frac = body[i] / rng[i]
        upper_frac = (h[i] - max(o[i], c[i])) / rng[i]
        lower_frac = (min(o[i], c[i]) - l[i]) / rng[i]
        if (label == "PRIOR_HIGH" and upper_frac >= 0.35 and c[i] < level) or (
            label == "PRIOR_LOW" and lower_frac >= 0.35 and c[i] > level
        ):
            rejection_count += 1
            event_type = "REJECTION_AT_PRIOR_LEVEL"

        # Pullback: the interaction candle moves against the recent local
        # direction before/at the level.
        recent_dir = np.sign(np.nanmean(direction[max(0, i - 3):i]))
        if recent_dir != 0 and direction[i] != 0 and direction[i] != recent_dir:
            pullback_count += 1
            event_type = "PULLBACK_TO_PRIOR_LEVEL"

        # Reaction/expansion: next candle, when available, expands its range.
        if i + 1 < len(w) and rng[i + 1] > np.nanmedian(rng[max(0, i - 4):i + 1]) * 1.25:
            reaction_expansion_count += 1

        events.append((i, event_type, label, source_idx, level))

    # Detect the dominant progression of candle size/direction around the last
    # interaction. This deliberately describes sequence behaviour only.
    story_parts = []
    if len(events):
        story_parts.append("PRIOR_LEVEL_INTERACTION")
    if pullback_count:
        story_parts.append("PULLBACK_PRESENT")
    if rejection_count:
        story_parts.append("REJECTION_PRESENT")
    if break_count:
        story_parts.append("LEVEL_BREAK_PRESENT")
    if hold_count:
        story_parts.append("LEVEL_HOLD_PRESENT")
    if repeated_level_hits:
        story_parts.append("REPEATED_LEVEL_TEST")
    if reaction_expansion_count:
        story_parts.append("POST_INTERACTION_EXPANSION")

    first_dir = float(np.nanmean(direction[:max(1, len(direction) // 3)]))
    last_dir = float(np.nanmean(direction[-max(1, len(direction) // 3):]))
    range_first = float(np.nanmean(rng[:max(1, len(rng) // 3)]))
    range_last = float(np.nanmean(rng[-max(1, len(rng) // 3):]))

    return {
        "interaction_count": len(events),
        "first_interaction_index": -1 if first_revisit is None else int(start + first_revisit),
        "last_interaction_index": -1 if last_interaction is None else int(start + last_interaction),
        "break_count": int(break_count),
        "hold_count": int(hold_count),
        "rejection_count": int(rejection_count),
        "pullback_count": int(pullback_count),
        "repeated_level_hits": int(repeated_level_hits),
        "reaction_expansion_count": int(reaction_expansion_count),
        "interaction_level_types": " -> ".join(interaction_types) if interaction_types else "NONE",
        "direction_first_third": first_dir,
        "direction_last_third": last_dir,
        "direction_shift": last_dir - first_dir,
        "range_first_third": range_first,
        "range_last_third": range_last,
        "range_shift": range_last - range_first,
        "story": " -> ".join(story_parts) if story_parts else "NO_PRIOR_LEVEL_INTERACTION",
    }


def _level_table(raw: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Return concrete prior levels that were available inside the window."""
    w = raw.iloc[start:end + 1].reset_index(drop=True)
    rows = []
    for i in range(1, len(w)):
        rows.append({
            "local_index": i,
            "open_time": int(w.iloc[i].open_time),
            "prior_high": float(w.iloc[i - 1].high),
            "prior_low": float(w.iloc[i - 1].low),
            "prior_body_high": float(max(w.iloc[i - 1].open, w.iloc[i - 1].close)),
            "prior_body_low": float(min(w.iloc[i - 1].open, w.iloc[i - 1].close)),
            "current_high": float(w.iloc[i].high),
            "current_low": float(w.iloc[i].low),
            "current_close": float(w.iloc[i].close),
        })
    return pd.DataFrame(rows)


def _plot_interaction(ax, raw: pd.DataFrame, start: int, end: int, title: str) -> None:
    w = raw.iloc[start:end + 1].reset_index(drop=True)
    x = np.arange(len(w), dtype=float)
    width = 0.60
    for i, row in w.iterrows():
        o, h, l, c = map(float, (row.open, row.high, row.low, row.close))
        ax.vlines(i, l, h, linewidth=0.8)
        lo = min(o, c)
        height = max(abs(c - o), EPS)
        ax.add_patch(plt.Rectangle((i - width / 2, lo), width, height, fill=c >= o, linewidth=0.7))
    for i in range(1, len(w)):
        # Draw immediately previous high/low as short horizontal reference marks.
        ax.hlines(float(w.iloc[i - 1].high), i - 0.42, i + 0.42, linewidth=0.55, alpha=0.45)
        ax.hlines(float(w.iloc[i - 1].low), i - 0.42, i + 0.42, linewidth=0.55, alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Candle position in window")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.12)


def main():
    p = argparse.ArgumentParser(description="Research ETH candle interaction with earlier price levels.")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--all-timeframes", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    timeframes = cfg.timeframes if args.all_timeframes else [args.timeframe]
    charts = Path("charts")
    charts.mkdir(exist_ok=True)
    for name in [
        "price_level_current.csv",
        "price_level_matches.csv",
        "price_level_events.csv",
        "price_level_map.png",
    ]:
        path = charts / name
        if path.exists():
            path.unlink()

    current_rows, match_rows, event_rows = [], [], []

    for tf in timeframes:
        print(f"[{tf}] loading and vectorizing candle behaviour...", flush=True)
        raw = load_market(cfg, tf)
        if len(raw) < args.window * 2:
            print(f"[{tf}] skipped: insufficient candles", flush=True)
            continue
        features = candle_features(raw)
        current_end = len(raw) - 1
        current_start = current_end - args.window + 1
        current_story = _interaction_for_window(raw, features, current_start, current_end)
        current_rows.append({"timeframe": tf, "window": args.window, **current_story})

        ranked = rank_candle_sequences(
            features,
            current_start=current_start,
            current_end=current_end,
            candidate_ends=range(args.window - 1, current_start),
            chunk_size=4096,
        ).head(args.top_k)
        print(f"[{tf}] ranked {len(ranked)} historical windows", flush=True)

        for rank, row in ranked.iterrows():
            s, e = int(row.candidate_start), int(row.candidate_end)
            story = _interaction_for_window(raw, features, s, e)
            match_rows.append({
                "timeframe": tf,
                "rank": rank + 1,
                "candidate_start": s,
                "candidate_end": e,
                "open_time": int(raw.iloc[s].open_time),
                "close_time": int(raw.iloc[e].close_time),
                "similarity": float(row.similarity),
                **story,
            })
            levels = _level_table(raw, s, e)
            for _, level in levels.iterrows():
                event_rows.append({
                    "timeframe": tf,
                    "rank": rank + 1,
                    "similarity": float(row.similarity),
                    "candidate_start": s,
                    "candidate_end": e,
                    **level.to_dict(),
                })

    current_df = pd.DataFrame(current_rows)
    matches_df = pd.DataFrame(match_rows)
    events_df = pd.DataFrame(event_rows)
    current_df.to_csv(charts / "price_level_current.csv", index=False)
    matches_df.to_csv(charts / "price_level_matches.csv", index=False)
    events_df.to_csv(charts / "price_level_events.csv", index=False)

    if not matches_df.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        for tf, g in matches_df.groupby("timeframe"):
            g = g.sort_values("rank")
            ax.plot(g["rank"], g["interaction_count"], marker="o", label=tf)
        ax.set_title("Price-Level Interaction Count in Historical Candle Sequences")
        ax.set_xlabel("Historical neighbour rank")
        ax.set_ylabel("Prior-level interactions")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "price_level_map.png", dpi=160)
        plt.close(fig)

    print("\nPRICE-LEVEL INTERACTION RESEARCH")
    print("Descriptive only: no entries, targets, stops, outcomes, or predictions.")
    print("\nCURRENT WINDOW STORIES")
    if not current_df.empty:
        cols = ["timeframe", "interaction_count", "break_count", "hold_count", "rejection_count", "pullback_count", "repeated_level_hits", "reaction_expansion_count", "story"]
        print(current_df[cols].to_string(index=False))

    print("\nTOP HISTORICAL WINDOWS")
    if not matches_df.empty:
        cols = ["timeframe", "rank", "similarity", "interaction_count", "break_count", "hold_count", "rejection_count", "pullback_count", "repeated_level_hits", "reaction_expansion_count", "story"]
        print(matches_df[cols].head(80).to_string(index=False))

    print("\nSaved outputs:")
    for name in ["price_level_current.csv", "price_level_matches.csv", "price_level_events.csv", "price_level_map.png"]:
        print(f"charts/{name}")


if __name__ == "__main__":
    main()
