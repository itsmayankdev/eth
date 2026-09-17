from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimilarityConfig:
    """Weights and scales used to compare two equal-length structures."""

    type_weight: float = 0.40
    swing_weight: float = 0.20
    leg_weight: float = 0.20
    duration_weight: float = 0.15
    shape_weight: float = 0.05
    swing_scale_pct: float = 2.0
    leg_scale_pct: float = 4.0
    duration_scale: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            self.type_weight,
            self.swing_weight,
            self.leg_weight,
            self.duration_weight,
            self.shape_weight,
        )
        if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("similarity weights must be non-negative and sum to 1")
        if self.swing_scale_pct <= 0 or self.leg_scale_pct <= 0 or self.duration_scale <= 0:
            raise ValueError("similarity scales must be greater than 0")


def _finite_numeric(values: pd.Series | np.ndarray, default: float = 0.0) -> np.ndarray:
    """Convert pandas/numpy numeric values, including pd.NA, into finite floats."""
    numeric = pd.to_numeric(values, errors="coerce")
    array = np.asarray(numeric, dtype=float)
    return np.nan_to_num(array, nan=default, posinf=default, neginf=default)


def _numeric_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Convert absolute numeric differences into a [0, 1] similarity."""
    if len(a) == 0:
        return 0.0
    diff = np.nan_to_num(
        np.abs(a - b),
        nan=scale * 10.0,
        posinf=scale * 10.0,
        neginf=scale * 10.0,
    )
    return float(np.exp(-np.mean(diff) / scale))


def _duration_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Compare pivot durations using a log-ratio so 2 vs 4 bars is meaningful."""
    a = np.maximum(np.nan_to_num(a, nan=1.0, posinf=1e9, neginf=1.0), 1e-9)
    b = np.maximum(np.nan_to_num(b, nan=1.0, posinf=1e9, neginf=1.0), 1e-9)
    diff = np.abs(np.log(a / b))
    return float(np.exp(-np.mean(diff) / scale))


def structure_similarity(
    current: pd.DataFrame,
    candidate: pd.DataFrame,
    config: SimilarityConfig | None = None,
) -> float:
    """Return a [0, 1] similarity score for two pivot sequences.

    Both sequences must contain the same number of pivots. The comparison uses
    structure labels, swing size, leg size, pivot duration, and normalized
    price shape. It does not use candles after either sequence's endpoint.
    """
    cfg = config or SimilarityConfig()
    required = {"pivot_type", "swing_pct", "leg_pct", "bars_since_prev", "normalized_price"}
    if not required.issubset(current.columns) or not required.issubset(candidate.columns):
        missing = sorted(required - set(current.columns) - set(candidate.columns))
        raise ValueError(f"missing structure columns: {missing}")
    if len(current) != len(candidate) or len(current) == 0:
        return 0.0

    left = current.reset_index(drop=True)
    right = candidate.reset_index(drop=True)

    type_similarity = float(
        np.mean(
            left["pivot_type"].astype("string").fillna("?").to_numpy()
            == right["pivot_type"].astype("string").fillna("?").to_numpy()
        )
    )
    swing_similarity = _numeric_similarity(
        _finite_numeric(left["swing_pct"]),
        _finite_numeric(right["swing_pct"]),
        cfg.swing_scale_pct,
    )
    leg_similarity = _numeric_similarity(
        _finite_numeric(left["leg_pct"]),
        _finite_numeric(right["leg_pct"]),
        cfg.leg_scale_pct,
    )
    duration_similarity = _duration_similarity(
        _finite_numeric(left["bars_since_prev"], default=1.0),
        _finite_numeric(right["bars_since_prev"], default=1.0),
        cfg.duration_scale,
    )

    # Re-anchor both shapes to their final pivot so absolute price level cannot
    # influence matching. The final pivot is therefore always exactly 1.0.
    left_shape = _finite_numeric(left["normalized_price"], default=1.0)
    right_shape = _finite_numeric(right["normalized_price"], default=1.0)
    left_shape = left_shape / max(abs(left_shape[-1]), 1e-12)
    right_shape = right_shape / max(abs(right_shape[-1]), 1e-12)
    shape_similarity = _numeric_similarity(left_shape, right_shape, 0.05)

    return float(
        cfg.type_weight * type_similarity
        + cfg.swing_weight * swing_similarity
        + cfg.leg_weight * leg_similarity
        + cfg.duration_weight * duration_similarity
        + cfg.shape_weight * shape_similarity
    )


def find_similar_structures(
    structure: pd.DataFrame,
    n_pivots: int = 8,
    top_k: int = 10,
    minimum_similarity: float = 0.0,
    config: SimilarityConfig | None = None,
    max_endpoint_index: int | None = None,
) -> pd.DataFrame:
    """Find historical matches ending before the current structure.

    The current structure is the final ``n_pivots`` rows of ``structure``.
    Candidate windows are required to end strictly before the current endpoint,
    which prevents matching the current event to itself and keeps the search
    causal. Results are sorted by descending similarity and then by recency.
    """
    columns = [
        "candidate_start_index",
        "candidate_end_index",
        "candidate_start_position",
        "candidate_end_position",
        "similarity",
        "pivot_type_sequence",
    ]
    if structure.empty or n_pivots <= 0 or top_k <= 0 or len(structure) < (2 * n_pivots):
        return pd.DataFrame(columns=columns)

    data = structure.reset_index(drop=True)
    current = data.tail(n_pivots).copy()
    current_end = int(current.iloc[-1]["index"])
    cutoff = current_end if max_endpoint_index is None else min(current_end, int(max_endpoint_index))

    matches: list[dict] = []
    # The latest historical candidate can end at any confirmed pivot strictly
    # before the current endpoint. Overlapping candidates are allowed because
    # this function is a retrieval primitive; walk-forward evaluation can add
    # stricter spacing rules later.
    for end_pos in range(n_pivots - 1, len(data) - n_pivots):
        candidate = data.iloc[end_pos - n_pivots + 1 : end_pos + 1].copy()
        candidate_end = int(candidate.iloc[-1]["index"])
        if candidate_end >= cutoff:
            continue

        score = structure_similarity(current, candidate, config)
        if score < minimum_similarity:
            continue
        matches.append(
            {
                "candidate_start_index": int(candidate.iloc[0]["index"]),
                "candidate_end_index": candidate_end,
                "candidate_start_position": int(end_pos - n_pivots + 1),
                "candidate_end_position": int(end_pos),
                "similarity": score,
                "pivot_type_sequence": " ".join(candidate["pivot_type"].astype("string").fillna("?").to_numpy()),
            }
        )

    if not matches:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(matches, columns=columns)
    return (
        result.sort_values(["similarity", "candidate_end_index"], ascending=[False, False])
        .head(top_k)
        .reset_index(drop=True)
    )
