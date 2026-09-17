from __future__ import annotations

"""Candle Story Engine — descriptive market-structure research only.

Combines candle behaviour and meaningful price-level interaction into an
ordered, human-readable story. It does not create trades, entries, targets,
stops, outcomes, or future-direction predictions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from data.database import MarketDatabase
from price_level_interaction_v2 import analyse_window, load_market, TIMEFRAMES

DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
MIN_STORY_SIMILARITY = 0.0

# Ordered story vocabulary. Earlier states are allowed to recur later; the
# engine preserves chronology rather than treating the window as a bag of tags.

def candle_state(o, h, l, c, prev_range):
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    body_frac = body / rng
    upper = (h - max(o, c)) / rng
    lower = (min(o, c) - l) / rng
    close_loc = (c - l) / rng
    range_ratio = rng / max(prev_range, 1e-12)
    if body_frac >= 0.60:
        body_state = "LARGE_BODY"
    elif body_frac <= 0.25:
        body_state = "SMALL_BODY"
    else:
        body_state = "MEDIUM_BODY"
    if range_ratio >= 1.35:
        range_state = "EXPANDING_RANGE"
    elif range_ratio <= 0.75:
        range_state = "COMPRESSING_RANGE"
    else:
        range_state = "NORMAL_RANGE"
    if upper >= 0.35 and lower < 0.25:
        wick_state = "UPPER_REJECTION"
    elif lower >= 0.35 and upper < 0.25:
        wick_state = "LOWER_REJECTION"
    elif upper >= 0.30 and lower >= 0.30:
        wick_state = "TWO_SIDED_REJECTION"
    else:
        wick_state = "BALANCED_WICK"
    if close_loc >= 0.65:
        close_state = "CLOSE_HIGH"
    elif close_loc <= 0.35:
        close_state = "CLOSE_LOW"
    else:
        close_state = "CLOSE_MID"
    direction = "BULLISH" if c > o else "BEARISH" if c < o else "NEUTRAL"
    return direction, body_state, range_state, wick_state, close_state


def build_story(df, start, end):
    w = df.iloc[start:end + 1].reset_index(drop=True)
    o = w.open.to_numpy(float); h = w.high.to_numpy(float)
    l = w.low.to_numpy(float); c = w.close.to_numpy(float)
    rng = np.maximum(h - l, 1e-12)
    events = []
    prev = float(np.median(rng[:max(1, min(5, len(rng)))]))

    # Price-level engine supplies the structural interaction events.
    level_row, level_events = analyse_window(df, start, end)
    level_by_i = {}
    for ev in level_events:
        level_by_i.setdefault(int(ev["local_index"]), []).append(ev)

    # One descriptive candle state per candle, then compress adjacent states
    # into meaningful phases. This keeps the final story readable.
    states = []
    for i in range(len(w)):
        state = candle_state(o[i], h[i], l[i], c[i], prev)
        states.append(state)
        prev = rng[i]

    for i, state in enumerate(states):
        direction, body_state, range_state, wick_state, close_state = state
        level_events_here = level_by_i.get(i, [])
        labels = []
        if i == 0:
            labels.append("WINDOW_START")
        if i > 0 and direction != states[i - 1][0] and direction != "NEUTRAL":
            labels.append("DIRECTION_CHANGE")
        if range_state == "EXPANDING_RANGE":
            labels.append("RANGE_EXPANSION")
        elif range_state == "COMPRESSING_RANGE":
            labels.append("RANGE_COMPRESSION")
        if body_state == "LARGE_BODY":
            labels.append("BODY_EXPANSION")
        if wick_state in {"UPPER_REJECTION", "LOWER_REJECTION", "TWO_SIDED_REJECTION"}:
            labels.append(wick_state)
        if close_state in {"CLOSE_HIGH", "CLOSE_LOW"}:
            labels.append(close_state)
        for ev in level_events_here:
            labels.append(ev["interaction"])
            if ev["repeated_test"]:
                labels.append("REPEATED_LEVEL_TEST")
            if ev["post_interaction_expansion"]:
                labels.append("POST_TEST_EXPANSION")
        if labels:
            events.append({
                "local_index": i,
                "open_time": int(w.iloc[i].open_time),
                "phase": " | ".join(labels),
                "direction": direction,
                "body_state": body_state,
                "range_state": range_state,
                "wick_state": wick_state,
                "close_state": close_state,
                "level_interaction_count": len(level_events_here),
            })

    # Phase compression: keep the first occurrence of a repeated phase family,
    # while preserving special structural events.
    phase_labels = []
    seen_families = set()
    for ev in events:
        family = ev["phase"].split(" | ")[0]
        structural = any(x in ev["phase"] for x in (
            "REJECTION", "HOLD", "BREAK", "PULLBACK", "REPEATED_LEVEL_TEST",
            "POST_TEST_EXPANSION", "DIRECTION_CHANGE"
        ))
        if structural or family not in seen_families:
            phase_labels.append(family)
            seen_families.add(family)

    direction = np.sign(c - o)
    split = len(w) // 2
    first_dir = float(np.mean(direction[:split])) if split else 0.0
    second_dir = float(np.mean(direction[split:])) if len(w) > split else 0.0
    range_shift = float(np.mean(rng[split:]) - np.mean(rng[:split])) if split else 0.0
    body_shift = float(np.mean(np.abs(c[split:] - o[split:])) - np.mean(np.abs(c[:split] - o[:split]))) if split else 0.0

    story = " -> ".join(phase_labels) if phase_labels else "NO_CANDLE_STORY"
    return {
        **level_row,
        "story": story,
        "story_event_count": len(events),
        "story_phase_count": len(phase_labels),
        "story_first_direction": first_dir,
        "story_second_direction": second_dir,
        "story_direction_shift": second_dir - first_dir,
        "story_range_shift": range_shift,
        "story_body_shift": body_shift,
    }, events


STORY_FEATURES = [
    "interaction_count", "break_count", "hold_count", "rejection_count",
    "pullback_count", "repeated_level_tests", "reaction_expansion_count",
    "swing_high_tests", "swing_low_tests", "body_zone_tests",
    "first_direction", "second_direction", "direction_shift",
    "first_range", "second_range", "range_shift", "first_body", "second_body",
    "body_shift", "first_close_location", "second_close_location",
    "close_location_shift", "first_interaction_position", "last_interaction_position",
    "level_count", "story_event_count", "story_phase_count", "story_first_direction",
    "story_second_direction", "story_direction_shift", "story_range_shift", "story_body_shift",
]


def rank_stories(current, candidates, top_k):
    x = np.asarray([[float(r.get(k, 0.0)) for k in STORY_FEATURES] for r in candidates], dtype=float)
    q25 = np.percentile(x, 25, axis=0); q75 = np.percentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-9)
    med = np.median(x, axis=0)
    z = (x - med) / scale
    q = np.asarray([float(current.get(k, 0.0)) for k in STORY_FEATURES])
    qz = (q - med) / scale
    d = np.linalg.norm(z - qz, axis=1) / np.sqrt(x.shape[1])
    sim = 1.0 / (1.0 + d)
    order = np.argsort(-sim)[:top_k]
    return [(int(i), float(sim[i])) for i in order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h", choices=TIMEFRAMES)
    p.add_argument("--all-timeframes", action="store_true")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = p.parse_args()

    cfg = load_config()
    tfs = TIMEFRAMES if args.all_timeframes else [args.timeframe]
    charts = Path("charts"); charts.mkdir(exist_ok=True)
    outputs = ["candle_story_current.csv", "candle_story_matches.csv", "candle_story_events.csv", "candle_story_map.png"]
    for name in outputs:
        (charts / name).unlink(missing_ok=True)

    current_rows, match_rows, event_rows = [], [], []
    for tf in tfs:
        df = load_market(cfg, tf)
        if len(df) < args.window * 2:
            print(f"[{tf}] skipped: insufficient history", flush=True); continue
        print(f"[{tf}] building candle stories ({len(df)} candles)...", flush=True)
        current_start = len(df) - args.window
        current, current_events = build_story(df, current_start, len(df) - 1)
        current_rows.append({"timeframe": tf, "window": args.window, **current})

        candidates = []
        for end in range(args.window - 1, current_start):
            row, _ = build_story(df, end - args.window + 1, end)
            candidates.append(row)
        ranked = rank_stories(current, candidates, args.top_k)
        print(f"[{tf}] ranked {len(ranked)} historical stories", flush=True)

        for rank, (ci, sim) in enumerate(ranked, 1):
            row = candidates[ci]
            s = args.window - 1 + ci - args.window + 1
            e = args.window - 1 + ci
            match_rows.append({
                "timeframe": tf, "rank": rank, "story_similarity": sim,
                "candidate_start": s, "candidate_end": e,
                "open_time": int(df.iloc[s].open_time), "close_time": int(df.iloc[e].close_time),
                **row,
            })
            _, ev = build_story(df, s, e)
            for event in ev:
                event_rows.append({"timeframe": tf, "rank": rank, "story_similarity": sim,
                                   "candidate_start": s, "candidate_end": e, **event})

    pd.DataFrame(current_rows).to_csv(charts / "candle_story_current.csv", index=False)
    pd.DataFrame(match_rows).to_csv(charts / "candle_story_matches.csv", index=False)
    pd.DataFrame(event_rows).to_csv(charts / "candle_story_events.csv", index=False)

    if match_rows:
        m = pd.DataFrame(match_rows)
        fig, ax = plt.subplots(figsize=(14, 7))
        for tf, g in m.groupby("timeframe"):
            g = g.sort_values("rank")
            ax.plot(g["rank"], g["story_similarity"], marker="o", label=tf)
        ax.set_title("Candle Story Similarity Across Historical Windows")
        ax.set_xlabel("Historical match rank"); ax.set_ylabel("Story similarity")
        ax.grid(alpha=0.15); ax.legend(); fig.tight_layout()
        fig.savefig(charts / "candle_story_map.png", dpi=160); plt.close(fig)

    print("\nCURRENT CANDLE STORIES")
    if current_rows:
        cols = ["timeframe", "story_event_count", "story_phase_count", "story"]
        print(pd.DataFrame(current_rows)[cols].to_string(index=False))
    print("\nSaved:")
    for name in outputs:
        print(f"charts/{name}")


if __name__ == "__main__":
    main()
