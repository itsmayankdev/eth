from __future__ import annotations

"""Candle Story Transition Map.

Research-only descriptive analysis of how compressed candle-story phases
follow one another in the historical match set. It does not create entries,
targets, stops, signals, forecasts, or trade recommendations.

Inputs:
    charts/candle_story_current.csv
    charts/candle_story_matches.csv

Outputs:
    charts/candle_story_transition_current.csv
    charts/candle_story_transition_pairs.csv
    charts/candle_story_transition_summary.csv
    charts/candle_story_transition_map.png
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_TOP_TRANSITIONS = 15


def parse_story(story: str) -> list[str]:
    if not isinstance(story, str) or not story.strip():
        return []
    return [x.strip() for x in story.split("->") if x.strip()]


def transitions(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else []


def transition_counts(stories: pd.Series) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for story in stories:
        counts.update(transitions(parse_story(story)))
    return counts


def transition_presence(stories: pd.Series) -> Counter[tuple[str, str]]:
    """Count how many distinct stories contain each transition."""
    counts: Counter[tuple[str, str]] = Counter()
    for story in stories:
        counts.update(set(transitions(parse_story(story))))
    return counts


def transition_rows(tf: str, stories: pd.Series, label: str) -> list[dict]:
    total = len(stories)
    occurrences = transition_counts(stories)
    presence = transition_presence(stories)
    rows = []
    for (a, b), count in occurrences.items():
        rows.append(
            {
                "timeframe": tf,
                "transition_from": a,
                "transition_to": b,
                "occurrences": count,
                "stories_with_transition": presence[(a, b)],
                "story_count": total,
                "story_presence_pct": (100.0 * presence[(a, b)] / total) if total else 0.0,
                "transition_label": f"{a} -> {b}",
                "source_set": label,
            }
        )
    return rows


def current_transition_rows(current: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in current.iterrows():
        tf = str(row["timeframe"])
        tokens = parse_story(row.get("story", ""))
        for idx, (a, b) in enumerate(transitions(tokens), 1):
            rows.append(
                {
                    "timeframe": tf,
                    "transition_order": idx,
                    "transition_from": a,
                    "transition_to": b,
                    "transition_label": f"{a} -> {b}",
                    "current_story": row.get("story", ""),
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-transitions", type=int, default=DEFAULT_TOP_TRANSITIONS)
    args = p.parse_args()

    charts = Path("charts")
    charts.mkdir(exist_ok=True)

    current_path = charts / "candle_story_current.csv"
    matches_path = charts / "candle_story_matches.csv"
    if not current_path.exists() or not matches_path.exists():
        raise FileNotFoundError("Run candle_story_engine.py first.")

    current = pd.read_csv(current_path)
    matches = pd.read_csv(matches_path)

    required = {"timeframe", "story", "rank"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns in candle_story_matches.csv: {sorted(missing)}")

    # Remove accidental duplicate rows for the same ranked match.
    matches = matches.drop_duplicates(subset=["timeframe", "rank"], keep="first").copy()

    current_rows = current_transition_rows(current)
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []

    for tf, group in matches.groupby("timeframe", sort=False):
        group = group.sort_values("rank")
        rows = transition_rows(str(tf), group["story"], "top_historical_matches")
        pair_rows.extend(rows)

        # Separate occurrence count from story presence. Presence is the
        # more useful descriptive measure because one story can repeat a pair.
        top = sorted(rows, key=lambda r: (-r["stories_with_transition"], -r["occurrences"], r["transition_label"]))
        for rank, item in enumerate(top[:args.top_transitions], 1):
            summary_rows.append({**item, "transition_rank": rank})

    current_df = pd.DataFrame(current_rows)
    pairs_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Annotate current transitions with historical presence when available.
    if not current_df.empty and not pairs_df.empty:
        lookup = pairs_df.drop_duplicates(
            ["timeframe", "transition_from", "transition_to"]
        ).set_index(["timeframe", "transition_from", "transition_to"])
        current_df["historical_story_presence_pct"] = [
            float(lookup.loc[(r.timeframe, r.transition_from, r.transition_to), "story_presence_pct"])
            if (r.timeframe, r.transition_from, r.transition_to) in lookup.index else 0.0
            for r in current_df.itertuples()
        ]
        current_df["historical_occurrences"] = [
            int(lookup.loc[(r.timeframe, r.transition_from, r.transition_to), "occurrences"])
            if (r.timeframe, r.transition_from, r.transition_to) in lookup.index else 0
            for r in current_df.itertuples()
        ]
        current_df["historical_transition_status"] = current_df["historical_story_presence_pct"].map(
            lambda x: "COMMON_IN_MATCH_SET" if x >= 50 else "PRESENT_IN_MATCH_SET" if x > 0 else "NOT_SEEN_IN_MATCH_SET"
        )

    current_df.to_csv(charts / "candle_story_transition_current.csv", index=False)
    pairs_df.to_csv(charts / "candle_story_transition_pairs.csv", index=False)
    summary_df.to_csv(charts / "candle_story_transition_summary.csv", index=False)

    # One readable chart: top transition presence by timeframe.
    if not summary_df.empty:
        plot_df = summary_df.sort_values(
            ["timeframe", "stories_with_transition", "occurrences"],
            ascending=[True, False, False],
        ).copy()
        plot_df["label"] = plot_df["timeframe"] + ": " + plot_df["transition_label"]
        plot_df = plot_df.sort_values("stories_with_transition", ascending=True).tail(40)

        fig, ax = plt.subplots(figsize=(15, 11))
        ax.barh(plot_df["label"], plot_df["stories_with_transition"])
        ax.set_xlabel("Historical matched stories containing transition")
        ax.set_ylabel("Timeframe and transition")
        ax.set_title("Candle Story Transition Presence")
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
        fig.savefig(charts / "candle_story_transition_map.png", dpi=160)
        plt.close(fig)

    print("\nCURRENT STORY TRANSITIONS")
    if current_df.empty:
        print("No transitions found")
    else:
        display_cols = [
            "timeframe", "transition_order", "transition_label",
            "historical_story_presence_pct", "historical_transition_status",
        ]
        print(current_df[display_cols].to_string(index=False))

    print("\nTOP HISTORICAL TRANSITIONS")
    if summary_df.empty:
        print("No historical transitions found")
    else:
        print(
            summary_df[
                [
                    "timeframe", "transition_rank", "transition_label",
                    "stories_with_transition", "story_count", "story_presence_pct",
                    "occurrences",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    for name in [
        "candle_story_transition_current.csv",
        "candle_story_transition_pairs.csv",
        "candle_story_transition_summary.csv",
        "candle_story_transition_map.png",
    ]:
        print(f"charts/{name}")


if __name__ == "__main__":
    main()
