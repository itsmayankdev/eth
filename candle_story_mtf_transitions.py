from __future__ import annotations

"""Multi-timeframe candle-story transition alignment.

Research-only descriptive analysis. It compares the current compressed story
phase sequence across all supported ETH/USDT timeframes. It does not create
entries, targets, stops, signals, forecasts, or trade recommendations.

Inputs:
    charts/candle_story_current.csv
    charts/candle_story_transition_current.csv
    charts/candle_mtf_history_matches.csv (optional)

Outputs:
    charts/candle_story_mtf_current.csv
    charts/candle_story_mtf_transition_pairs.csv
    charts/candle_story_mtf_transition_matrix.csv
    charts/candle_story_mtf_map.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TIMEFRAME_ORDER = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_GROUP_GAP = 2


def parse_story(story: str) -> list[str]:
    if not isinstance(story, str) or not story.strip():
        return []
    return [x.strip() for x in story.split("->") if x.strip()]


def story_transitions(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else []


def transition_label(a: str, b: str) -> str:
    return f"{a} -> {b}"


def phase_bucket(token: str) -> str:
    """Map detailed phases into broad behavioural families for cross-TF comparison."""
    if token.startswith("INITIAL_"):
        return "INITIAL_CONTEXT"
    if "RANGE_" in token:
        return "RANGE_CHANGE"
    if token in {"DIRECTIONAL_SHIFT", "DIRECTION_CHANGE_IN_CHARACTER", "COUNTERMOVE_AND_RENEWAL"}:
        return "DIRECTION_CHANGE"
    if "BODY_" in token:
        return "BODY_CHANGE"
    if token in {"LEVEL_INTERACTION_CLUSTER", "REPEATED_TEST_CLUSTER"}:
        return "LEVEL_INTERACTION"
    if token == "REJECTION_CLUSTER":
        return "REJECTION"
    if token == "PULLBACK_CLUSTER":
        return "PULLBACK"
    if token == "HOLD_CLUSTER":
        return "HOLD"
    if token == "EXPANSION_AFTER_TEST":
        return "EXPANSION_AFTER_TEST"
    return token


def compress_consecutive(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        if not out or out[-1] != token:
            out.append(token)
    return out


def common_transitions(stories: dict[str, list[str]]) -> list[dict]:
    rows = []
    for tf in TIMEFRAME_ORDER:
        tokens = stories.get(tf, [])
        seen = set(story_transitions(tokens))
        for a, b in sorted(seen):
            rows.append(
                {
                    "timeframe": tf,
                    "transition_from": a,
                    "transition_to": b,
                    "transition_label": transition_label(a, b),
                    "phase_from_family": phase_bucket(a),
                    "phase_to_family": phase_bucket(b),
                }
            )
    return rows


def family_sequence(tokens: list[str]) -> list[str]:
    return compress_consecutive([phase_bucket(x) for x in tokens])


def alignment_score(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return max(0.0, 1.0 - dp[n][m] / max(n, m))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=DEFAULT_GROUP_GAP)
    args = parser.parse_args()

    charts = Path("charts")
    current_path = charts / "candle_story_current.csv"
    transition_path = charts / "candle_story_transition_current.csv"

    if not current_path.exists():
        raise FileNotFoundError("Run candle_story_engine.py first.")
    if not transition_path.exists():
        raise FileNotFoundError("Run candle_story_transitions.py first.")

    current = pd.read_csv(current_path)
    current = current[current["timeframe"].isin(TIMEFRAME_ORDER)].copy()

    stories = {
        str(row.timeframe): parse_story(row.story)
        for row in current.itertuples()
    }

    rows = []
    for tf in TIMEFRAME_ORDER:
        tokens = stories.get(tf, [])
        families = family_sequence(tokens)
        rows.append(
            {
                "timeframe": tf,
                "phase_count": len(tokens),
                "family_phase_count": len(families),
                "story": " -> ".join(tokens),
                "family_story": " -> ".join(families),
            }
        )

    current_df = pd.DataFrame(rows)

    # Pairwise cross-timeframe alignment of broad story families.
    pair_rows = []
    for i, tf_a in enumerate(TIMEFRAME_ORDER):
        if tf_a not in stories:
            continue
        for tf_b in TIMEFRAME_ORDER[i + 1 :]:
            if tf_b not in stories:
                continue
            a = family_sequence(stories[tf_a])
            b = family_sequence(stories[tf_b])
            score = alignment_score(a, b)
            pair_rows.append(
                {
                    "timeframe_a": tf_a,
                    "timeframe_b": tf_b,
                    "alignment_score": score,
                    "family_sequence_a": " -> ".join(a),
                    "family_sequence_b": " -> ".join(b),
                    "relationship": (
                        "HIGH_FAMILY_ALIGNMENT" if score >= 0.75
                        else "MODERATE_FAMILY_ALIGNMENT" if score >= 0.50
                        else "LOW_FAMILY_ALIGNMENT"
                    ),
                }
            )

    pairs_df = pd.DataFrame(pair_rows)

    # Transition-family matrix: how many timeframes currently contain each
    # broad transition family.
    family_pairs = defaultdict(list)
    for tf, tokens in stories.items():
        fam = family_sequence(tokens)
        for a, b in story_transitions(fam):
            family_pairs[(a, b)].append(tf)

    matrix_rows = []
    for (a, b), tfs in sorted(family_pairs.items()):
        row = {
            "transition_family": f"{a} -> {b}",
            "timeframe_count": len(tfs),
            "timeframes": ",".join(tfs),
        }
        for tf in TIMEFRAME_ORDER:
            row[tf] = 1 if tf in tfs else 0
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)

    current_df.to_csv(charts / "candle_story_mtf_current.csv", index=False)
    pairs_df.to_csv(charts / "candle_story_mtf_transition_pairs.csv", index=False)
    matrix_df.to_csv(charts / "candle_story_mtf_transition_matrix.csv", index=False)

    if not pairs_df.empty:
        pivot = pairs_df.pivot(
            index="timeframe_a",
            columns="timeframe_b",
            values="alignment_score",
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Second timeframe")
        ax.set_ylabel("First timeframe")
        ax.set_title("Cross-Timeframe Candle Story Family Alignment")
        fig.colorbar(im, ax=ax, label="Family-sequence alignment")
        fig.tight_layout()
        fig.savefig(charts / "candle_story_mtf_map.png", dpi=160)
        plt.close(fig)

    print("\nCURRENT MULTI-TIMEFRAME STORY FAMILIES")
    print(current_df[["timeframe", "family_phase_count", "family_story"]].to_string(index=False))

    print("\nCROSS-TIMEFRAME PAIR ALIGNMENT")
    if pairs_df.empty:
        print("No timeframe pairs available")
    else:
        print(
            pairs_df[
                ["timeframe_a", "timeframe_b", "alignment_score", "relationship"]
            ].to_string(index=False)
        )

    print("\nTRANSITION FAMILIES SHARED ACROSS TIMEFRAMES")
    if matrix_df.empty:
        print("No transition families found")
    else:
        print(matrix_df.to_string(index=False))

    print("\nSaved:")
    for name in [
        "candle_story_mtf_current.csv",
        "candle_story_mtf_transition_pairs.csv",
        "candle_story_mtf_transition_matrix.csv",
        "candle_story_mtf_map.png",
    ]:
        print(f"charts/{name}")


if __name__ == "__main__":
    main()
