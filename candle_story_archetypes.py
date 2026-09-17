from __future__ import annotations

"""Recurring multi-timeframe candle-story archetype discovery.

Research-only descriptive analysis. Groups synchronized historical story
families into recurring structural archetypes. No entries, targets, stops,
signals, forecasts, or trade recommendations.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_TOP_K = 10

FAMILIES = [
    "INITIAL_CONTEXT", "RANGE_CHANGE", "DIRECTION_CHANGE", "BODY_CHANGE",
    "LEVEL_INTERACTION", "REJECTION", "PULLBACK", "HOLD", "EXPANSION_AFTER_TEST"
]


def normalize_story(value: str) -> list[str]:
    return [x.strip() for x in str(value).split("->") if x.strip()]


def compress(tokens: list[str]) -> list[str]:
    out = []
    for x in tokens:
        if x in FAMILIES and (not out or out[-1] != x):
            out.append(x)
    return out


def story_distance(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 1.0
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if a[i - 1] == b[j - 1] else 1.0
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m] / max(n, m))


def family_vector(stories: dict[str, list[str]]) -> np.ndarray:
    v = []
    for tf in TIMEFRAMES:
        s = stories.get(tf, [])
        for f in FAMILIES:
            v.append(s.count(f) / max(1, len(s)))
    return np.asarray(v, dtype=float)


def transition_pairs(stories: dict[str, list[str]]) -> set[tuple[str, str, str]]:
    out = set()
    for tf in TIMEFRAMES:
        s = stories.get(tf, [])
        for a, b in zip(s, s[1:]):
            out.add((tf, a, b))
    return out


def pair_similarity(a, b) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_archetype_id(index: int) -> str:
    return f"ARCH-{index:02d}"


def run(input_file: str, top_k: int, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    df = pd.read_csv(input_file)
    if df.empty:
        raise RuntimeError("No historical replay rows found")

    story_cols = [f"{tf}_story" for tf in TIMEFRAMES if f"{tf}_story" in df.columns]
    if not story_cols:
        raise RuntimeError("No timeframe story columns found")

    records = []
    for _, row in df.iterrows():
        stories = {}
        for tf in TIMEFRAMES:
            col = f"{tf}_story"
            if col in df.columns:
                stories[tf] = compress(normalize_story(row[col]))
        records.append(stories)

    # Deterministic greedy archetype discovery. A new archetype is created
    # only when it is structurally farther than the threshold from existing
    # representatives. Threshold is descriptive, not predictive.
    threshold = 0.48
    archetypes = []
    assignments = []
    for idx, stories in enumerate(records):
        if not archetypes:
            archetypes.append([idx, stories])
            assignments.append(0)
            continue
        distances = []
        pairs = transition_pairs(stories)
        for _, rep in archetypes:
            ds = []
            for tf in TIMEFRAMES:
                ds.append(story_distance(stories.get(tf, []), rep.get(tf, [])))
            d_story = float(np.mean(ds))
            d_pairs = 1.0 - pair_similarity(pairs, transition_pairs(rep))
            distances.append(0.7 * d_story + 0.3 * d_pairs)
        best = int(np.argmin(distances))
        if distances[best] > threshold:
            archetypes.append([idx, stories])
            assignments.append(len(archetypes) - 1)
        else:
            assignments.append(best)

    summary = []
    members = []
    for aid, (_, rep) in enumerate(archetypes):
        inds = [i for i, a in enumerate(assignments) if a == aid]
        summary.append({
            "archetype": make_archetype_id(aid + 1),
            "occurrence_count": len(inds),
            "occurrence_share": len(inds) / len(records),
            "representative_index": inds[0],
            "representative_timestamp": int(df.iloc[inds[0]]["reference_timestamp"]),
            "representative_combined_similarity": float(df.iloc[inds[0]].get("combined_similarity", np.nan)),
            "description": "Recurring descriptive multi-timeframe story structure",
        })
        for i in inds:
            members.append({
                "archetype": make_archetype_id(aid + 1),
                "member_index": i,
                "reference_timestamp": int(df.iloc[i]["reference_timestamp"]),
                "rank": int(df.iloc[i].get("rank", i + 1)),
                "combined_similarity": float(df.iloc[i].get("combined_similarity", np.nan)),
            })

    summary_df = pd.DataFrame(summary).sort_values("occurrence_count", ascending=False).reset_index(drop=True)
    members_df = pd.DataFrame(members).sort_values(["archetype", "member_index"])

    # Representative story table.
    rep_rows = []
    for s in summary:
        i = int(s["representative_index"])
        row = df.iloc[i]
        r = {"archetype": s["archetype"], "reference_timestamp": int(row["reference_timestamp"])}
        for tf in TIMEFRAMES:
            col = f"{tf}_story"
            if col in df.columns:
                r[col] = " -> ".join(records[i].get(tf, []))
        rep_rows.append(r)
    reps_df = pd.DataFrame(rep_rows)

    # Pair recurrence across all historical synchronized states.
    pair_counts = {}
    for stories in records:
        for p in transition_pairs(stories):
            pair_counts[p] = pair_counts.get(p, 0) + 1
    pair_rows = [
        {"timeframe": tf, "from_family": a, "to_family": b, "occurrence_count": c, "occurrence_share": c / len(records)}
        for (tf, a, b), c in pair_counts.items()
    ]
    pair_df = pd.DataFrame(pair_rows).sort_values("occurrence_count", ascending=False)

    summary_df.to_csv(out / "candle_story_archetype_summary.csv", index=False)
    members_df.to_csv(out / "candle_story_archetype_members.csv", index=False)
    reps_df.to_csv(out / "candle_story_archetype_representatives.csv", index=False)
    pair_df.to_csv(out / "candle_story_archetype_transitions.csv", index=False)

    # Heatmap: archetype x timeframe story length.
    rows = []
    for aid, (_, rep) in enumerate(archetypes):
        inds = [i for i, a in enumerate(assignments) if a == aid]
        for tf in TIMEFRAMES:
            vals = [len(records[i].get(tf, [])) for i in inds]
            rows.append({"archetype": make_archetype_id(aid + 1), "timeframe": tf, "mean_phase_count": np.mean(vals) if vals else 0})
    heat = pd.DataFrame(rows).pivot(index="archetype", columns="timeframe", values="mean_phase_count")
    fig, ax = plt.subplots(figsize=(12, max(5, len(heat) * 0.45)))
    im = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto")
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
    ax.set_xlabel("Timeframe"); ax.set_ylabel("Archetype"); ax.set_title("Recurring Candle Story Archetypes — Mean Phase Count")
    fig.colorbar(im, ax=ax, label="Mean compressed phases")
    fig.tight_layout(); fig.savefig(out / "candle_story_archetype_map.png", dpi=160); plt.close(fig)

    print("\nRECURRING STORY ARCHETYPES")
    print(summary_df.head(top_k).to_string(index=False))
    print("\nMOST RECURRENT CROSS-TIMEFRAME STORY TRANSITIONS")
    print(pair_df.head(20).to_string(index=False))
    print("\nSaved:")
    for name in [
        "candle_story_archetype_summary.csv",
        "candle_story_archetype_members.csv",
        "candle_story_archetype_representatives.csv",
        "candle_story_archetype_transitions.csv",
        "candle_story_archetype_map.png",
    ]:
        print(f"charts/{name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="charts/candle_story_mtf_history_matches_1h.csv")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--output-dir", default="charts")
    args = p.parse_args()
    run(args.input, args.top_k, args.output_dir)
