from __future__ import annotations

"""Refined price-level interaction research.

Research-only. No entries, targets, stops, outcomes, or predictions.

The key refinement is that a level must first exist in the *anchor half* of a
window. The reaction half is then evaluated only against those meaningful
levels. This avoids counting every candle as an interaction with every earlier
candle.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from data.database import MarketDatabase

TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
TOLERANCE_RANGE = 0.20
MIN_SEPARATION_ATR = 0.50
EXPANSION_MULT = 1.25

FEATURES = [
    "interaction_count", "break_count", "hold_count", "rejection_count",
    "pullback_count", "repeated_level_tests", "reaction_expansion_count",
    "swing_high_tests", "swing_low_tests", "body_zone_tests",
    "first_direction", "second_direction", "direction_shift",
    "first_range", "second_range", "range_shift", "first_body",
    "second_body", "body_shift", "first_close_location", "second_close_location",
    "close_location_shift", "first_interaction_position", "last_interaction_position",
]


def load_market(cfg, tf):
    db = MarketDatabase(cfg.database)
    try:
        df = db.load_candles(cfg.symbol, tf)
    finally:
        db.close()
    if df.empty:
        raise ValueError(f"No data for {tf}")
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def robust_atr(h, l, c):
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return np.maximum(pd.Series(tr).rolling(14, min_periods=3).median().bfill().to_numpy(), 1e-12)


def local_levels(o, h, l, c, atr, split):
    levels = []
    for i in range(1, max(1, split - 1)):
        if h[i] >= h[i - 1] and h[i] >= h[i + 1]:
            levels.append({"price": float(h[i]), "kind": "SWING_HIGH", "anchor": i})
        if l[i] <= l[i - 1] and l[i] <= l[i + 1]:
            levels.append({"price": float(l[i]), "kind": "SWING_LOW", "anchor": i})
    body = np.abs(c[:split] - o[:split])
    threshold = np.median(body) * 1.25 if len(body) else 0.0
    for i in range(split):
        if body[i] > 0 and body[i] >= threshold:
            levels.append({"price": float((o[i] + c[i]) / 2), "kind": "BODY_ZONE", "anchor": i})

    # Merge nearby levels so several candles around the same price become one
    # meaningful zone instead of many duplicate interactions.
    merged = []
    for level in sorted(levels, key=lambda x: x["price"]):
        tol = max(float(atr[min(level["anchor"], len(atr) - 1)]) * MIN_SEPARATION_ATR, 1e-12)
        found = None
        for existing in merged:
            if abs(level["price"] - existing["price"]) <= tol:
                found = existing
                break
        if found is None:
            merged.append({**level, "kinds": {level["kind"]}})
        else:
            found["price"] = (found["price"] + level["price"]) / 2
            found["kinds"].add(level["kind"])
            found["anchor"] = min(found["anchor"], level["anchor"])
    return merged


def analyse_window(df, start, end):
    w = df.iloc[start:end + 1].reset_index(drop=True)
    o = w.open.to_numpy(float); h = w.high.to_numpy(float)
    l = w.low.to_numpy(float); c = w.close.to_numpy(float)
    v = w.volume.to_numpy(float)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    direction = np.sign(c - o)
    close_loc = (c - l) / rng
    atr = robust_atr(h, l, c)
    split = len(w) // 2
    levels = local_levels(o, h, l, c, atr, split)

    events = []
    tested = set()
    breaks = holds = rejects = pulls = repeats = expansions = 0
    sh_tests = sl_tests = body_tests = 0

    first_dir = float(np.mean(direction[:split]))
    second_dir = float(np.mean(direction[split:]))
    first_range = float(np.mean(rng[:split])); second_range = float(np.mean(rng[split:]))
    first_body = float(np.mean(body[:split])); second_body = float(np.mean(body[split:]))
    first_cl = float(np.mean(close_loc[:split])); second_cl = float(np.mean(close_loc[split:]))

    for i in range(split, len(w)):
        candidates = []
        for li, level in enumerate(levels):
            tol = max(atr[i] * TOLERANCE_RANGE, 1e-12)
            distance = min(abs(h[i] - level["price"]), abs(l[i] - level["price"]), abs(c[i] - level["price"]))
            if distance <= tol:
                candidates.append((distance / tol, li, level))
        if not candidates:
            continue
        _, li, level = min(candidates, key=lambda x: x[0])
        p = level["price"]; tol = max(atr[i] * TOLERANCE_RANGE, 1e-12)
        repeated = li in tested
        tested.add(li)
        repeats += int(repeated)
        kinds = level["kinds"]
        sh_tests += int("SWING_HIGH" in kinds)
        sl_tests += int("SWING_LOW" in kinds)
        body_tests += int("BODY_ZONE" in kinds)

        close_above = c[i] > p + tol
        close_below = c[i] < p - tol
        crossed = (l[i] < p - tol and close_above) or (h[i] > p + tol and close_below)
        if crossed:
            breaks += 1
            event = "BREAK"
        else:
            event = "TEST"

        upper = (h[i] - max(o[i], c[i])) / rng[i]
        lower = (min(o[i], c[i]) - l[i]) / rng[i]
        rejection = ("SWING_HIGH" in kinds or "BODY_ZONE" in kinds) and upper >= 0.35 and c[i] < p
        rejection = rejection or (("SWING_LOW" in kinds or "BODY_ZONE" in kinds) and lower >= 0.35 and c[i] > p)
        if rejection:
            rejects += 1
            event = "REJECTION"

        if ("SWING_HIGH" in kinds and c[i] <= p + tol) or ("SWING_LOW" in kinds and c[i] >= p - tol):
            holds += 1
            if event == "TEST":
                event = "HOLD"

        recent = np.sign(np.mean(direction[max(0, i - 3):i])) if i > split else 0
        if recent != 0 and direction[i] != 0 and direction[i] != recent:
            pulls += 1
            if event == "TEST":
                event = "PULLBACK"

        expanded = False
        if i + 1 < len(w):
            baseline = np.median(rng[max(split, i - 4):i + 1])
            expanded = bool(rng[i + 1] > baseline * EXPANSION_MULT)
            expansions += int(expanded)

        events.append({
            "local_index": i,
            "open_time": int(w.iloc[i].open_time),
            "level_price": float(p),
            "level_type": "+".join(sorted(kinds)),
            "interaction": event,
            "repeated_test": int(repeated),
            "post_interaction_expansion": int(expanded),
            "open": float(o[i]), "high": float(h[i]), "low": float(l[i]),
            "close": float(c[i]), "volume": float(v[i]),
        })

    n = max(len(w) - split, 1)
    row = {
        "interaction_count": len(events),
        "break_count": breaks,
        "hold_count": holds,
        "rejection_count": rejects,
        "pullback_count": pulls,
        "repeated_level_tests": repeats,
        "reaction_expansion_count": expansions,
        "swing_high_tests": sh_tests,
        "swing_low_tests": sl_tests,
        "body_zone_tests": body_tests,
        "first_direction": first_dir,
        "second_direction": second_dir,
        "direction_shift": second_dir - first_dir,
        "first_range": first_range,
        "second_range": second_range,
        "range_shift": second_range - first_range,
        "first_body": first_body,
        "second_body": second_body,
        "body_shift": second_body - first_body,
        "first_close_location": first_cl,
        "second_close_location": second_cl,
        "close_location_shift": second_cl - first_cl,
        "first_interaction_position": events[0]["local_index"] / n if events else -1,
        "last_interaction_position": events[-1]["local_index"] / n if events else -1,
        "level_count": len(levels),
    }
    story = []
    if events: story.append("MEANINGFUL_LEVEL_TEST")
    if pulls: story.append("PULLBACK_TO_LEVEL")
    if rejects: story.append("REJECTION")
    if breaks: story.append("BREAK")
    if holds: story.append("HOLD")
    if repeats: story.append("REPEATED_TEST")
    if expansions: story.append("POST_TEST_EXPANSION")
    row["story"] = " -> ".join(story) if story else "NO_MEANINGFUL_LEVEL_INTERACTION"
    return row, events


def feature_vector(row):
    return np.asarray([float(row.get(k, 0.0)) for k in FEATURES], dtype=float)


def rank_candidates(current, candidates, top_k):
    x = np.vstack([feature_vector(current)] + [feature_vector(r) for r in candidates])
    med = np.median(x, axis=0); q25 = np.percentile(x, 25, axis=0); q75 = np.percentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-9)
    z = (x - med) / scale
    d = np.linalg.norm(z[1:] - z[0], axis=1) / np.sqrt(x.shape[1])
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

    cfg = load_config(); tfs = TIMEFRAMES if args.all_timeframes else [args.timeframe]
    charts = Path("charts"); charts.mkdir(exist_ok=True)
    for name in ("price_level_current.csv", "price_level_matches.csv", "price_level_events.csv", "price_level_map.png"):
        (charts / name).unlink(missing_ok=True)

    current_rows = []; match_rows = []; event_rows = []
    for tf in tfs:
        df = load_market(cfg, tf)
        if len(df) < args.window * 2:
            print(f"[{tf}] skipped: insufficient history"); continue
        print(f"[{tf}] preparing meaningful-level windows ({len(df)} candles)...", flush=True)
        current_start = len(df) - args.window
        current, current_events = analyse_window(df, current_start, len(df) - 1)
        current_rows.append({"timeframe": tf, "window": args.window, **current})

        candidates = []
        for end in range(args.window - 1, current_start):
            row, _ = analyse_window(df, end - args.window + 1, end)
            candidates.append(row)
        ranked = rank_candidates(current, candidates, args.top_k)
        print(f"[{tf}] ranked {len(ranked)} historical windows", flush=True)

        for rank, (ci, sim) in enumerate(ranked, 1):
            row = candidates[ci]
            s = current_start if False else (args.window - 1 + ci) - args.window + 1
            e = args.window - 1 + ci
            match_rows.append({
                "timeframe": tf, "rank": rank, "similarity": sim,
                "candidate_start": s, "candidate_end": e,
                "open_time": int(df.iloc[s].open_time), "close_time": int(df.iloc[e].close_time),
                **row,
            })
            _, ev = analyse_window(df, s, e)
            for event in ev:
                event_rows.append({"timeframe": tf, "rank": rank, "similarity": sim, "candidate_start": s, "candidate_end": e, **event})

    pd.DataFrame(current_rows).to_csv(charts / "price_level_current.csv", index=False)
    pd.DataFrame(match_rows).to_csv(charts / "price_level_matches.csv", index=False)
    pd.DataFrame(event_rows).to_csv(charts / "price_level_events.csv", index=False)

    if match_rows:
        m = pd.DataFrame(match_rows)
        fig, ax = plt.subplots(figsize=(14, 7))
        for tf, g in m.groupby("timeframe"):
            g = g.sort_values("rank")
            ax.plot(g["rank"], g["interaction_count"], marker="o", label=tf)
        ax.set_title("Meaningful Price-Level Interaction Across Historical Matches")
        ax.set_xlabel("Historical match rank"); ax.set_ylabel("Interaction count")
        ax.grid(alpha=0.15); ax.legend(); fig.tight_layout()
        fig.savefig(charts / "price_level_map.png", dpi=160); plt.close(fig)

    print("\nCURRENT WINDOW")
    if current_rows:
        cols = ["timeframe", "level_count", "interaction_count", "break_count", "hold_count", "rejection_count", "pullback_count", "repeated_level_tests", "reaction_expansion_count", "story"]
        print(pd.DataFrame(current_rows)[cols].to_string(index=False))
    print("\nSaved:")
    for n in ("price_level_current.csv", "price_level_matches.csv", "price_level_events.csv", "price_level_map.png"):
        print(f"charts/{n}")


if __name__ == "__main__":
    main()
