from __future__ import annotations

"""Candle Story Engine — descriptive market-structure research only.

The engine has two stages:
1) vectorized candle-behaviour screening across the full history;
2) detailed price-level analysis only for the strongest candidate windows.

This avoids running the expensive price-level detector over every historical
window. The output is a compact chronological story, not a trading signal.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from price_level_interaction_v2 import analyse_window, load_market, TIMEFRAMES

DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
SCREEN_MULTIPLIER = 4

CANDLE_FEATURES = [
    "first_direction", "second_direction", "direction_shift",
    "first_range", "second_range", "range_shift",
    "first_body", "second_body", "body_shift",
    "first_close_location", "second_close_location", "close_location_shift",
    "first_alternation", "second_alternation",
    "first_small_body", "second_small_body",
    "first_large_body", "second_large_body",
    "first_upper_rejection", "second_upper_rejection",
    "first_lower_rejection", "second_lower_rejection",
]

STORY_FEATURES = [
    "interaction_count", "break_count", "hold_count", "rejection_count",
    "pullback_count", "repeated_level_tests", "reaction_expansion_count",
    "swing_high_tests", "swing_low_tests", "body_zone_tests",
    "first_direction", "second_direction", "direction_shift",
    "first_range", "second_range", "range_shift", "first_body", "second_body",
    "body_shift", "first_close_location", "second_close_location",
    "close_location_shift", "first_interaction_position", "last_interaction_position",
    "level_count",
]


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cs[window:] - cs[:-window]) / float(window)


def build_screen_features(df: pd.DataFrame, window: int) -> tuple[np.ndarray, list[str]]:
    o = df.open.to_numpy(float)
    h = df.high.to_numpy(float)
    l = df.low.to_numpy(float)
    c = df.close.to_numpy(float)
    n = len(df)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    direction = np.sign(c - o)
    close_loc = (c - l) / rng
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    alt = np.r_[0.0, (direction[1:] != direction[:-1]).astype(float)]
    small = (body / rng <= 0.25).astype(float)
    large = (body / rng >= 0.60).astype(float)
    upper_rej = ((upper >= 0.35) & (lower < 0.25)).astype(float)
    lower_rej = ((lower >= 0.35) & (upper < 0.25)).astype(float)

    m = n - window + 1
    if m <= 0:
        return np.empty((0, len(CANDLE_FEATURES))), CANDLE_FEATURES

    # Window halves are calculated from prefix sums, so every historical
    # window is represented without a Python loop.
    half = window // 2

    def win_mean(x):
        return rolling_mean(x, window)

    def half_mean(x, second=False):
        cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
        starts = np.arange(m)
        a = starts + (half if second else 0)
        b = starts + (window if second else half)
        return (cs[b] - cs[a]) / float(half)

    first_dir = half_mean(direction)
    second_dir = half_mean(direction, True)
    first_range = half_mean(rng)
    second_range = half_mean(rng, True)
    first_body = half_mean(body)
    second_body = half_mean(body, True)
    first_cl = half_mean(close_loc)
    second_cl = half_mean(close_loc, True)
    first_alt = half_mean(alt)
    second_alt = half_mean(alt, True)
    first_small = half_mean(small)
    second_small = half_mean(small, True)
    first_large = half_mean(large)
    second_large = half_mean(large, True)
    first_upper = half_mean(upper_rej)
    second_upper = half_mean(upper_rej, True)
    first_lower = half_mean(lower_rej)
    second_lower = half_mean(lower_rej, True)

    matrix = np.column_stack([
        first_dir, second_dir, second_dir - first_dir,
        first_range, second_range, second_range - first_range,
        first_body, second_body, second_body - first_body,
        first_cl, second_cl, second_cl - first_cl,
        first_alt, second_alt,
        first_small, second_small,
        first_large, second_large,
        first_upper, second_upper,
        first_lower, second_lower,
    ])
    return matrix, CANDLE_FEATURES


def robust_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    x = np.vstack([query, candidates])
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-9)
    z = (x - med) / scale
    d = np.linalg.norm(z[1:] - z[0], axis=1) / np.sqrt(x.shape[1])
    return 1.0 / (1.0 + d)


def major_candle_phases(df: pd.DataFrame, start: int, end: int) -> list[str]:
    w = df.iloc[start:end + 1]
    o = w.open.to_numpy(float)
    h = w.high.to_numpy(float)
    l = w.low.to_numpy(float)
    c = w.close.to_numpy(float)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    direction = np.sign(c - o)
    split = len(w) // 2

    phases: list[str] = []
    d1 = float(np.mean(direction[:split]))
    d2 = float(np.mean(direction[split:]))
    r1 = float(np.mean(rng[:split])); r2 = float(np.mean(rng[split:]))
    b1 = float(np.mean(body[:split])); b2 = float(np.mean(body[split:]))

    if d1 >= 0.25:
        phases.append("INITIAL_BULLISH_BIAS")
    elif d1 <= -0.25:
        phases.append("INITIAL_BEARISH_BIAS")
    else:
        phases.append("INITIAL_BALANCED_BEHAVIOUR")

    if r2 > r1 * 1.20:
        phases.append("RANGE_EXPANSION")
    elif r2 < r1 * 0.80:
        phases.append("RANGE_COMPRESSION")

    if d1 * d2 < -0.03 and abs(d2 - d1) >= 0.35:
        phases.append("DIRECTIONAL_SHIFT")
    elif abs(d2 - d1) >= 0.35:
        phases.append("DIRECTION_CHANGE_IN_CHARACTER")

    if b2 > b1 * 1.20:
        phases.append("BODY_EXPANSION")
    elif b2 < b1 * 0.80:
        phases.append("BODY_COMPRESSION")

    # Detect a compact pullback-like middle move without calling it a signal.
    if len(direction) >= 6:
        third = len(direction) // 3
        a = float(np.mean(direction[:third]))
        m = float(np.mean(direction[third:2 * third]))
        z = float(np.mean(direction[2 * third:]))
        if a * m < -0.05 and z * a > 0.02:
            phases.append("COUNTERMOVE_AND_RENEWAL")

    return phases


def build_story(df: pd.DataFrame, start: int, end: int):
    level_row, level_events = analyse_window(df, start, end)
    phases = major_candle_phases(df, start, end)

    # Price-level events are the structural backbone of the story. We retain
    # their chronology but collapse duplicate event types.
    ordered = []
    seen = set()
    for ev in sorted(level_events, key=lambda x: int(x["local_index"])):
        name = str(ev["interaction"])
        if name not in seen:
            ordered.append(name)
            seen.add(name)
        if int(ev["repeated_test"]):
            if "REPEATED_LEVEL_TEST" not in seen:
                ordered.append("REPEATED_LEVEL_TEST"); seen.add("REPEATED_LEVEL_TEST")
        if int(ev["post_interaction_expansion"]):
            if "POST_TEST_EXPANSION" not in seen:
                ordered.append("POST_TEST_EXPANSION"); seen.add("POST_TEST_EXPANSION")

    # Translate raw interaction vocabulary into a compact structural story.
    structural_map = {
        "TEST": "LEVEL_TEST",
        "PULLBACK": "PULLBACK_TO_LEVEL",
        "REJECTION": "REJECTION",
        "HOLD": "HOLD",
        "BREAK": "LEVEL_BREAK",
    }
    interaction_story = []
    for x in ordered:
        interaction_story.append(structural_map.get(x, x))

    # Only retain candle phases that add information not already expressed by
    # the level lifecycle. This prevents a story from becoming one event per candle.
    story = []
    for phase in phases:
        if phase not in story:
            story.append(phase)
    for phase in interaction_story:
        if phase not in story:
            story.append(phase)

    # A level lifecycle is more informative than a generic candle descriptor,
    # so put it after the broad context but preserve its chronological order.
    if not story:
        story = ["NO_MEANINGFUL_STORY"]

    row = {
        **level_row,
        "story": " -> ".join(story),
        "story_event_count": len(level_events),
        "story_phase_count": len(story),
    }
    return row, level_events


def detailed_rank(current: dict, candidates: list[dict], top_k: int):
    if not candidates:
        return []
    x = np.asarray([[float(r.get(k, 0.0)) for k in STORY_FEATURES] for r in candidates], dtype=float)
    q = np.asarray([float(current.get(k, 0.0)) for k in STORY_FEATURES], dtype=float)
    sim = robust_similarity(q, x)
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
            print(f"[{tf}] skipped: insufficient history", flush=True)
            continue

        print(f"[{tf}] vectorized candle screening ({len(df)} candles)...", flush=True)
        screen, _ = build_screen_features(df, args.window)
        current_idx = len(df) - args.window
        current_screen = screen[-1]
        candidate_screen = screen[:-1]

        # First pass: cheap NumPy screening. Only a small pool receives the
        # expensive price-level analysis.
        coarse_sim = robust_similarity(current_screen, candidate_screen)
        pool_size = min(len(coarse_sim), max(args.top_k * SCREEN_MULTIPLIER, 100))
        coarse_order = np.argsort(-coarse_sim)[:pool_size]

        current, current_events = build_story(df, current_idx, len(df) - 1)
        current_rows.append({"timeframe": tf, "window": args.window, **current})

        candidates = []
        candidate_indices = []
        for screen_idx in coarse_order:
            end = int(screen_idx) + args.window - 1
            start = end - args.window + 1
            row, _ = build_story(df, start, end)
            row["screen_similarity"] = float(coarse_sim[screen_idx])
            candidates.append(row)
            candidate_indices.append((start, end))

        ranked = detailed_rank(current, candidates, args.top_k)
        print(f"[{tf}] screened {len(candidate_screen):,} windows -> detailed {len(candidates)} -> ranked {len(ranked)}", flush=True)

        for rank, (ci, sim) in enumerate(ranked, 1):
            row = candidates[ci]
            s, e = candidate_indices[ci]
            match_rows.append({
                "timeframe": tf,
                "rank": rank,
                "story_similarity": sim,
                "screen_similarity": row.get("screen_similarity", 0.0),
                "candidate_start": s,
                "candidate_end": e,
                "open_time": int(df.iloc[s].open_time),
                "close_time": int(df.iloc[e].close_time),
                **row,
            })
            _, ev = build_story(df, s, e)
            for event in ev:
                event_rows.append({
                    "timeframe": tf, "rank": rank,
                    "story_similarity": sim,
                    "candidate_start": s, "candidate_end": e,
                    **event,
                })

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
        ax.set_xlabel("Historical match rank")
        ax.set_ylabel("Story similarity")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "candle_story_map.png", dpi=160)
        plt.close(fig)

    print("\nCURRENT CANDLE STORIES")
    if current_rows:
        cols = ["timeframe", "story_event_count", "story_phase_count", "story"]
        print(pd.DataFrame(current_rows)[cols].to_string(index=False))
    print("\nSaved:")
    for name in outputs:
        print(f"charts/{name}")


if __name__ == "__main__":
    main()
