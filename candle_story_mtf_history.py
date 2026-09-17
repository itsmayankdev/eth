from __future__ import annotations

"""Historical synchronized multi-timeframe Candle Story Replay.

Research-only descriptive analysis. It finds exact common completed-candle
reference timestamps across ETH/USDT timeframes, screens synchronized history
with vectorized candle-structure similarity, then reconstructs compressed
chronological stories for a small candidate pool.

No entries, targets, stops, signals, forecasts, or trade recommendations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from candle_story_engine import DEFAULT_WINDOW, analyse_window, raw_story_tokens
from price_level_interaction_v2 import load_market
from candle_mtf_history import TIMEFRAMES, REFERENCE_TF, TF_MS, candle_features, build_history, feature_vector, similarity

DEFAULT_TOP_K = 25
SCREEN_POOL = 100


def family_token(token: str) -> str:
    if token.startswith("INITIAL_"): return "INITIAL_CONTEXT"
    if "RANGE_" in token: return "RANGE_CHANGE"
    if token in {"DIRECTIONAL_SHIFT", "DIRECTION_CHANGE_IN_CHARACTER", "COUNTERMOVE_AND_RENEWAL"}: return "DIRECTION_CHANGE"
    if "BODY_" in token: return "BODY_CHANGE"
    if token in {"LEVEL_INTERACTION_CLUSTER", "REPEATED_TEST_CLUSTER"}: return "LEVEL_INTERACTION"
    if token == "REJECTION_CLUSTER": return "REJECTION"
    if token == "PULLBACK_CLUSTER": return "PULLBACK"
    if token == "HOLD_CLUSTER": return "HOLD"
    if token == "EXPANSION_AFTER_TEST": return "EXPANSION_AFTER_TEST"
    return token


def family_sequence(tokens: list[str]) -> list[str]:
    out = []
    for token in tokens:
        f = family_token(token)
        if not out or out[-1] != f: out.append(f)
    return out


def seq_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b: return 0.0
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1, dtype=float)
    dp[0, :] = np.arange(m + 1, dtype=float)
    related = {frozenset(x) for x in [
        ("LEVEL_INTERACTION", "REJECTION"),
        ("LEVEL_INTERACTION", "PULLBACK"),
        ("REJECTION", "HOLD"),
        ("RANGE_CHANGE", "BODY_CHANGE"),
        ("DIRECTION_CHANGE", "BODY_CHANGE"),
        ("LEVEL_INTERACTION", "EXPANSION_AFTER_TEST"),
    ]}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            x, y = a[i - 1], b[j - 1]
            cost = 0.0 if x == y else 0.5 if frozenset((x, y)) in related else 1.0
            dp[i, j] = min(dp[i - 1, j] + 1.0, dp[i, j - 1] + 1.0, dp[i - 1, j - 1] + cost)
    return max(0.0, 1.0 - dp[n, m] / max(n, m))


def exact_index(closes: np.ndarray, ts: int) -> int | None:
    j = int(np.searchsorted(closes, ts, side="left"))
    return j if j < len(closes) and int(closes[j]) == int(ts) else None


def build_story(df: pd.DataFrame, start: int, end: int) -> list[str]:
    # analyse_window returns exactly (summary, events).
    _, events = analyse_window(df, start, end)
    return raw_story_tokens(df, start, end, events)


def run(window: int, top_k: int, screen_pool: int, output_dir: str) -> None:
    if window < 4 or window % 2: raise ValueError("window must be an even number >= 4")
    out = Path(output_dir); out.mkdir(exist_ok=True)
    cfg = load_config()
    data = {}

    for tf in TIMEFRAMES:
        df = load_market(cfg, tf)
        if len(df) < window: continue
        df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        feats = candle_features(df)
        matrix, starts = build_history(feats, window)
        current = feature_vector(feats, len(feats) - window, window)
        sims = similarity(current, matrix)
        ends = starts + window - 1
        opens = df.iloc[ends]["open_time"].to_numpy(dtype=np.int64)
        closes = opens + TF_MS[tf]
        order = np.argsort(closes)
        data[tf] = {"df": df, "starts": starts[order], "ends": ends[order], "opens": opens[order], "closes": closes[order], "similarity": sims[order]}
        print(f"[{tf}] {len(matrix):,} synchronized-window candidates prepared")

    if len(data) != len(TIMEFRAMES): raise RuntimeError("All eight timeframes are required")

    ref = data[REFERENCE_TF]
    common_start = max(int(x["closes"][0]) for x in data.values())
    common_end = min(int(x["closes"][-1]) for x in data.values())
    reference_times = ref["closes"][(ref["closes"] >= common_start) & (ref["closes"] <= common_end)]
    reference_times = reference_times[reference_times != int(ref["closes"][-1])]

    exact = []
    for ts in reference_times:
        ts = int(ts); indices = {}; ok = True
        for tf in TIMEFRAMES:
            j = exact_index(data[tf]["closes"], ts)
            if j is None: ok = False; break
            indices[tf] = j
        if ok: exact.append((ts, indices))
    if not exact: raise RuntimeError("No exact synchronized historical timestamps found")
    print(f"Exact synchronized historical timestamps: {len(exact):,}")

    screened = []
    for ts, indices in exact:
        scores = [data[tf]["similarity"][indices[tf]] for tf in TIMEFRAMES]
        screened.append((float(np.mean(scores)), ts, indices))
    screened.sort(key=lambda x: x[0], reverse=True)
    screened = screened[:max(top_k, screen_pool)]

    current_path = out / "candle_story_current.csv"
    if current_path.exists():
        current_df = pd.read_csv(current_path)
        current_stories = {str(r.timeframe): str(r.story) for r in current_df.itertuples()}
    else:
        current_stories = {}
        for tf in TIMEFRAMES:
            item = data[tf]
            current_stories[tf] = " -> ".join(build_story(item["df"], int(item["starts"][-1]), int(item["ends"][-1])))
    current_family = {tf: family_sequence(story.split(" -> ")) for tf, story in current_stories.items()}

    rows = []; story_rows = []
    for screen_rank, (feature_score, ts, indices) in enumerate(screened, 1):
        story_scores = []; tf_stories = {}
        for tf in TIMEFRAMES:
            item = data[tf]; j = indices[tf]
            start = int(item["starts"][j]); end = int(item["ends"][j])
            tokens = build_story(item["df"], start, end); fam = family_sequence(tokens)
            tf_stories[tf] = tokens
            score = seq_similarity(current_family[tf], fam); story_scores.append(score)
            story_rows.append({"reference_timestamp": ts, "screen_rank": screen_rank, "timeframe": tf, "candidate_start": start, "candidate_end": end, "candle_structure_similarity": float(item["similarity"][j]), "story_similarity": float(score), "family_story": " -> ".join(fam), "story": " -> ".join(tokens)})
        story_score = float(np.mean(story_scores)); combined = 0.60 * feature_score + 0.40 * story_score
        rows.append({"reference_timestamp": ts, "screen_rank": screen_rank, "combined_similarity": combined, "cross_tf_candle_similarity": feature_score, "cross_tf_story_similarity": story_score, "alignment_status": "EXACTLY_SYNCHRONIZED", "max_alignment_gap_ms": 0, **{f"{tf}_story": " -> ".join(tf_stories[tf]) for tf in TIMEFRAMES}})

    results = pd.DataFrame(rows).sort_values("combined_similarity", ascending=False).head(top_k).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    detail = pd.DataFrame(story_rows)
    selected_refs = set(results["reference_timestamp"].astype(int))
    detail = detail[detail["reference_timestamp"].astype(int).isin(selected_refs)].copy()

    current_out = pd.DataFrame([{"timeframe": tf, "current_story": current_stories[tf], "current_family_story": " -> ".join(current_family[tf]), "research_note": "Exact synchronized completed-candle reference; descriptive only"} for tf in TIMEFRAMES])
    results.to_csv(out / "candle_story_mtf_history_matches.csv", index=False)
    detail.to_csv(out / "candle_story_mtf_history_matrix.csv", index=False)
    current_out.to_csv(out / "candle_story_mtf_history_current.csv", index=False)

    if not detail.empty:
        heat = detail.pivot_table(index="reference_timestamp", columns="timeframe", values="story_similarity", aggfunc="mean").reindex(columns=TIMEFRAMES)
        fig, ax = plt.subplots(figsize=(13, 10)); im = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns)
        ax.set_yticks(range(len(heat.index))); ax.set_yticklabels([str(x) for x in heat.index])
        ax.set_xlabel("Timeframe"); ax.set_ylabel("Exact synchronized reference timestamp"); ax.set_title("Historical Synchronized Candle Story Similarity")
        fig.colorbar(im, ax=ax, label="Story-family similarity"); fig.tight_layout(); fig.savefig(out / "candle_story_mtf_history_map.png", dpi=160); plt.close(fig)

    print("\nCURRENT SYNCHRONIZED STORY FAMILIES")
    print(current_out.to_string(index=False))
    print("\nTOP EXACTLY SYNCHRONIZED HISTORICAL STORY REPLAYS")
    print(results[["rank", "reference_timestamp", "combined_similarity", "cross_tf_candle_similarity", "cross_tf_story_similarity", "max_alignment_gap_ms"]].to_string(index=False))
    print("\nTOP REPLAY STORIES BY TIMEFRAME")
    for _, r in results.head(min(5, len(results))).iterrows():
        print(f"\nRANK {int(r['rank'])} | timestamp={int(r['reference_timestamp'])} | combined={r['combined_similarity']:.4f}")
        for tf in TIMEFRAMES: print(f"  {tf}: {r[f'{tf}_story']}")
    print("\nSaved:")
    for name in ["candle_story_mtf_history_current.csv", "candle_story_mtf_history_matches.csv", "candle_story_mtf_history_matrix.csv", "candle_story_mtf_history_map.png"]: print(f"charts/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--screen-pool", type=int, default=SCREEN_POOL)
    parser.add_argument("--output-dir", default="charts")
    args = parser.parse_args()
    run(args.window, args.top_k, args.screen_pool, args.output_dir)
