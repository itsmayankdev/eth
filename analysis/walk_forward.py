from __future__ import annotations

from math import ceil
from statistics import NormalDist

import numpy as np
import pandas as pd

from analysis.outcomes import OutcomeConfig, evaluate_matches, measure_match_outcome, summarize_outcomes
from structure.similarity import find_similar_structures_at


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (np.nan, np.nan)
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    phat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denominator
    margin = z * np.sqrt((phat * (1.0 - phat) / trials) + (z2 / (4.0 * trials * trials))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_outcomes_with_ci(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Add Wilson intervals for all-sample and decisive target rates."""
    summary = summarize_outcomes(outcomes, horizons)
    if summary.empty:
        return summary

    summary = summary.copy()
    all_lowers: list[float] = []
    all_uppers: list[float] = []
    decisive_lowers: list[float] = []
    decisive_uppers: list[float] = []

    for _, row in summary.iterrows():
        lo, hi = wilson_interval(int(row["target"]), int(row["samples"]), confidence_level)
        decisive_n = int(row["target"] + row["stop"])
        d_lo, d_hi = wilson_interval(int(row["target"]), decisive_n, confidence_level)
        all_lowers.append(lo)
        all_uppers.append(hi)
        decisive_lowers.append(d_lo)
        decisive_uppers.append(d_hi)

    summary["target_rate_all_lower"] = all_lowers
    summary["target_rate_all_upper"] = all_uppers
    summary["target_rate_decisive_lower"] = decisive_lowers
    summary["target_rate_decisive_upper"] = decisive_uppers
    return summary


def _sample_endpoint_positions(
    structure: pd.DataFrame,
    candles: pd.DataFrame,
    n_pivots: int,
    max_samples: int,
    spacing_candles: int,
    max_horizon: int,
) -> list[int]:
    """Select historical endpoints spaced far enough to avoid overlapping outcomes."""
    if max_samples <= 0 or spacing_candles <= 0:
        return []
    eligible: list[int] = []
    for pos in range(2 * n_pivots - 1, len(structure)):
        confirmation = int(structure.iloc[pos]["confirmation_index"])
        if confirmation + max_horizon >= len(candles):
            continue
        eligible.append(pos)

    if len(eligible) <= max_samples:
        candidates = eligible
    else:
        stride = max(1, ceil(len(eligible) / max_samples))
        candidates = eligible[::stride]
        if len(candidates) > max_samples:
            candidates = candidates[:max_samples]

    selected: list[int] = []
    last_confirmation = -10**18
    for pos in candidates:
        confirmation = int(structure.iloc[pos]["confirmation_index"])
        if confirmation - last_confirmation >= spacing_candles:
            selected.append(pos)
            last_confirmation = confirmation
    return selected


def evaluate_walk_forward(
    structure: pd.DataFrame,
    candles: pd.DataFrame,
    n_pivots: int = 8,
    top_k: int = 10,
    minimum_similarity: float = 0.70,
    outcome_config: OutcomeConfig | None = None,
    max_samples: int = 250,
    spacing_candles: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated historical matching with strictly-forward outcomes.

    Each endpoint uses only pivots confirmed by that endpoint. Its top-K
    historical matches are evaluated after their own confirmation candle.
    Endpoint samples are spaced by at least the maximum outcome horizon by
    default, reducing overlapping forward windows in the aggregate.
    """
    cfg = outcome_config or OutcomeConfig()
    max_horizon = max(int(h) for h in cfg.horizons)
    spacing = max_horizon if spacing_candles is None else int(spacing_candles)
    positions = _sample_endpoint_positions(
        structure,
        candles,
        n_pivots,
        max_samples,
        spacing,
        max_horizon,
    )

    all_matches: list[pd.DataFrame] = []
    for endpoint_pos in positions:
        matches = find_similar_structures_at(
            structure,
            current_end_position=endpoint_pos,
            n_pivots=n_pivots,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )
        if matches.empty:
            continue
        outcomes = evaluate_matches(matches, candles, structure, cfg)
        if outcomes.empty:
            continue
        outcomes["evaluation_endpoint_position"] = endpoint_pos
        outcomes["evaluation_endpoint_index"] = int(structure.iloc[endpoint_pos]["index"])
        outcomes["evaluation_endpoint_confirmation_index"] = int(structure.iloc[endpoint_pos]["confirmation_index"])
        all_matches.append(outcomes)

    if not all_matches:
        return pd.DataFrame(), pd.DataFrame()

    result = pd.concat(all_matches, ignore_index=True)
    summary = summarize_outcomes_with_ci(result, cfg.horizons)
    return result, summary


def evaluate_endpoint_baseline(
    structure: pd.DataFrame,
    candles: pd.DataFrame,
    evaluation_endpoint_positions: list[int],
    outcome_config: OutcomeConfig,
) -> pd.DataFrame:
    """Measure the actual forward outcome at each evaluation endpoint.

    This is the no-matching baseline: one trade per evaluation endpoint,
    entered at that endpoint's confirmation close and using the same barriers
    and horizons as the historical-match evaluation.
    """
    closes = pd.to_numeric(candles["close"], errors="coerce").to_numpy(float)
    rows: list[dict] = []
    for pos in evaluation_endpoint_positions:
        if pos < 0 or pos >= len(structure):
            continue
        confirmation = int(structure.iloc[pos]["confirmation_index"])
        if confirmation < 0 or confirmation >= len(closes) or not np.isfinite(closes[confirmation]):
            continue
        outcome = measure_match_outcome(
            candles,
            confirmation,
            float(closes[confirmation]),
            outcome_config.horizons,
            outcome_config.target_pct,
            outcome_config.stop_pct,
        )
        outcome["evaluation_endpoint_position"] = pos
        outcome["evaluation_endpoint_index"] = int(structure.iloc[pos]["index"])
        outcome["entry_index"] = confirmation
        rows.append(outcome)
    return pd.DataFrame(rows)


def deoverlap_outcomes(
    outcomes: pd.DataFrame,
    max_horizon: int,
) -> pd.DataFrame:
    """Keep a non-overlapping subset based on entry candles.

    Rows are ordered by entry time. Once an outcome is kept, another row is
    accepted only when its entry candle is at least ``max_horizon`` candles
    later. This removes repeated/overlapping forward windows from aggregate
    statistics without using future outcome information to select rows.
    """
    if outcomes.empty or max_horizon <= 0 or "entry_index" not in outcomes.columns:
        return outcomes.copy()
    ordered = outcomes.sort_values(["entry_index", "evaluation_endpoint_position", "similarity"], ascending=[True, True, False])
    keep: list[int] = []
    last_entry = -10**18
    for idx, row in ordered.iterrows():
        entry = int(row["entry_index"])
        if entry - last_entry >= max_horizon:
            keep.append(idx)
            last_entry = entry
    return ordered.loc[keep].reset_index(drop=True)


def summarize_similarity_buckets(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
    confidence_level: float = 0.95,
    bins: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 1.01),
) -> pd.DataFrame:
    """Summarize outcomes by similarity range to test whether strength matters."""
    if outcomes.empty or "similarity" not in outcomes.columns:
        return pd.DataFrame()
    labels = [f"{bins[i]:.2f}–{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    work = outcomes.copy()
    work["similarity_bucket"] = pd.cut(
        pd.to_numeric(work["similarity"], errors="coerce"),
        bins=list(bins),
        labels=labels,
        right=False,
        include_lowest=True,
    )
    rows: list[dict] = []
    for label in labels:
        bucket = work[work["similarity_bucket"] == label]
        if bucket.empty:
            continue
        for h in horizons:
            col = f"target_first_{int(h)}"
            if col not in bucket:
                continue
            values = bucket[col].dropna().astype(str)
            decisive = values[values.isin(["target", "stop"])]
            target = int((values == "target").sum())
            samples = len(values)
            d_n = len(decisive)
            lo, hi = wilson_interval(target, samples, confidence_level)
            d_lo, d_hi = wilson_interval(target, d_n, confidence_level)
            rows.append({
                "similarity_bucket": label,
                "horizon": int(h),
                "samples": samples,
                "target": target,
                "stop": int((values == "stop").sum()),
                "neither": int((values == "neither").sum()),
                "ambiguous": int((values == "ambiguous").sum()),
                "target_rate_all": target / samples if samples else np.nan,
                "target_rate_all_lower": lo,
                "target_rate_all_upper": hi,
                "target_rate_decisive": target / d_n if d_n else np.nan,
                "target_rate_decisive_lower": d_lo,
                "target_rate_decisive_upper": d_hi,
            })
    return pd.DataFrame(rows)
