from __future__ import annotations

from math import ceil, log

import numpy as np
import pandas as pd

from analysis.outcomes import OutcomeConfig, evaluate_matches, summarize_outcomes
from structure.similarity import find_similar_structures_at


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (np.nan, np.nan)
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    # Normal quantile for the requested two-sided confidence level.
    # Statistics/normal libraries are intentionally avoided to keep dependencies small.
    p = (1.0 + confidence_level) / 2.0
    a = log(1.0 / (1.0 - p))
    b = log(1.0 / p)
    z = np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
    z = float(z)

    phat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denominator
    margin = z * np.sqrt((phat * (1.0 - phat) / trials) + (z2 / (4.0 * trials * trials))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _erfinv(x: float) -> float:
    """Approximate inverse error function using Winitzki's formula."""
    x = float(np.clip(x, -0.999999, 0.999999))
    a = 0.147
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    return float(np.sign(x) * np.sqrt(np.sqrt(first * first - ln / a) - first))


def summarize_outcomes_with_ci(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Add Wilson intervals for both all-sample and decisive target rates."""
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
        # Start with evenly distributed candidates, then enforce spacing in time.
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
    """Run repeated historical structure matching with strictly-forward outcomes.

    Each evaluation endpoint uses only pivots confirmed by that endpoint. Its
    top-K historical matches are evaluated after their own confirmation candle.
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
