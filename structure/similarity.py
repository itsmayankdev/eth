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
        weights = (self.type_weight, self.swing_weight, self.leg_weight, self.duration_weight, self.shape_weight)
        if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("similarity weights must be non-negative and sum to 1")
        if self.swing_scale_pct <= 0 or self.leg_scale_pct <= 0 or self.duration_scale <= 0:
            raise ValueError("similarity scales must be greater than 0")


def _finite_numeric(values: pd.Series | np.ndarray, default: float = 0.0) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    array = np.asarray(numeric, dtype=float)
    return np.nan_to_num(array, nan=default, posinf=default, neginf=default)


def _numeric_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    if len(a) == 0:
        return 0.0
    diff = np.nan_to_num(np.abs(a - b), nan=scale * 10.0, posinf=scale * 10.0, neginf=scale * 10.0)
    return float(np.exp(-np.mean(diff) / scale))


def _duration_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    a = np.maximum(np.nan_to_num(a, nan=1.0, posinf=1e9, neginf=1.0), 1e-9)
    b = np.maximum(np.nan_to_num(b, nan=1.0, posinf=1e9, neginf=1.0), 1e-9)
    return float(np.exp(-np.mean(np.abs(np.log(a / b))) / scale))


def structure_similarity(current: pd.DataFrame, candidate: pd.DataFrame, config: SimilarityConfig | None = None) -> float:
    """Return the similarity score for two equal-length pivot sequences."""
    cfg = config or SimilarityConfig()
    required = {"pivot_type", "swing_pct", "leg_pct", "bars_since_prev", "normalized_price"}
    if not required.issubset(current.columns) or not required.issubset(candidate.columns):
        missing = sorted(required - set(current.columns) - set(candidate.columns))
        raise ValueError(f"missing structure columns: {missing}")
    if len(current) != len(candidate) or len(current) == 0:
        return 0.0

    left = current.reset_index(drop=True)
    right = candidate.reset_index(drop=True)
    type_similarity = float(np.mean(left["pivot_type"].astype("string").fillna("?").to_numpy() == right["pivot_type"].astype("string").fillna("?").to_numpy()))
    swing_similarity = _numeric_similarity(_finite_numeric(left["swing_pct"]), _finite_numeric(right["swing_pct"]), cfg.swing_scale_pct)
    leg_similarity = _numeric_similarity(_finite_numeric(left["leg_pct"]), _finite_numeric(right["leg_pct"]), cfg.leg_scale_pct)
    duration_similarity = _duration_similarity(_finite_numeric(left["bars_since_prev"], 1.0), _finite_numeric(right["bars_since_prev"], 1.0), cfg.duration_scale)
    left_shape = _finite_numeric(left["normalized_price"], 1.0)
    right_shape = _finite_numeric(right["normalized_price"], 1.0)
    left_shape = left_shape / max(abs(left_shape[-1]), 1e-12)
    right_shape = right_shape / max(abs(right_shape[-1]), 1e-12)
    shape_similarity = _numeric_similarity(left_shape, right_shape, 0.05)
    return float(cfg.type_weight * type_similarity + cfg.swing_weight * swing_similarity + cfg.leg_weight * leg_similarity + cfg.duration_weight * duration_similarity + cfg.shape_weight * shape_similarity)


def _fast_candidate_scores(current: pd.DataFrame, data: pd.DataFrame, end_positions: np.ndarray, n_pivots: int, config: SimilarityConfig) -> np.ndarray:
    """Vectorized scoring across candidate windows; formula matches structure_similarity."""
    current_types = current["pivot_type"].astype("string").fillna("?").to_numpy()
    current_swing = _finite_numeric(current["swing_pct"])
    current_leg = _finite_numeric(current["leg_pct"])
    current_duration = _finite_numeric(current["bars_since_prev"], 1.0)
    current_shape = _finite_numeric(current["normalized_price"], 1.0)
    current_shape = current_shape / max(abs(current_shape[-1]), 1e-12)

    types = data["pivot_type"].astype("string").fillna("?").to_numpy()
    swings = _finite_numeric(data["swing_pct"])
    legs = _finite_numeric(data["leg_pct"])
    durations = np.maximum(_finite_numeric(data["bars_since_prev"], 1.0), 1e-9)
    shapes = _finite_numeric(data["normalized_price"], 1.0)

    starts = end_positions - n_pivots + 1
    idx = starts[:, None] + np.arange(n_pivots)[None, :]
    type_similarity = np.mean(types[idx] == current_types[None, :], axis=1)
    swing_similarity = np.exp(-np.mean(np.abs(swings[idx] - current_swing[None, :]), axis=1) / config.swing_scale_pct)
    duration_similarity = np.exp(-np.mean(np.abs(np.log(durations[idx] / current_duration[None, :])), axis=1) / config.duration_scale)
    leg_similarity = np.exp(-np.mean(np.abs(legs[idx] - current_leg[None, :]), axis=1) / config.leg_scale_pct)
    candidate_shape = shapes[idx]
    candidate_shape = candidate_shape / np.maximum(np.abs(candidate_shape[:, -1:]), 1e-12)
    shape_similarity = np.exp(-np.mean(np.abs(candidate_shape - current_shape[None, :]), axis=1) / 0.05)

    return (config.type_weight * type_similarity + config.swing_weight * swing_similarity + config.leg_weight * leg_similarity + config.duration_weight * duration_similarity + config.shape_weight * shape_similarity)


def find_similar_structures_at(structure: pd.DataFrame, current_end_position: int, n_pivots: int = 8, top_k: int = 10, minimum_similarity: float = 0.0, config: SimilarityConfig | None = None, max_endpoint_index: int | None = None) -> pd.DataFrame:
    """Find historical matches for a known endpoint using vectorized candidate scoring."""
    columns = ["candidate_start_index", "candidate_end_index", "candidate_start_position", "candidate_end_position", "candidate_confirmation_index", "similarity", "pivot_type_sequence"]
    if structure.empty or current_end_position < 0 or current_end_position >= len(structure) or n_pivots <= 0 or top_k <= 0:
        return pd.DataFrame(columns=columns)

    data = structure.reset_index(drop=True)
    if current_end_position + 1 < 2 * n_pivots:
        return pd.DataFrame(columns=columns)

    current = data.iloc[current_end_position - n_pivots + 1:current_end_position + 1].copy()
    current_confirmation = int(current.iloc[-1]["confirmation_index"])
    cutoff = current_confirmation if max_endpoint_index is None else min(current_confirmation, int(max_endpoint_index))
    end_positions = np.arange(n_pivots - 1, current_end_position - n_pivots + 1, dtype=int)
    if len(end_positions) == 0:
        return pd.DataFrame(columns=columns)

    confirmations = _finite_numeric(data["confirmation_index"], default=np.nan)[end_positions]
    valid = confirmations < cutoff
    if not np.any(valid):
        return pd.DataFrame(columns=columns)
    end_positions = end_positions[valid]
    confirmations = confirmations[valid].astype(int)

    cfg = config or SimilarityConfig()
    scores = _fast_candidate_scores(current, data, end_positions, n_pivots, cfg)
    keep = scores >= minimum_similarity
    if not np.any(keep):
        return pd.DataFrame(columns=columns)
    end_positions, confirmations, scores = end_positions[keep], confirmations[keep], scores[keep]

    indices = data["index"].to_numpy()
    order = np.lexsort((indices[end_positions], -scores))[:top_k]
    end_positions, confirmations, scores = end_positions[order], confirmations[order], scores[order]
    starts = end_positions - n_pivots + 1
    types = data["pivot_type"].astype("string").fillna("?").to_numpy()

    rows = []
    for start_pos, end_pos, confirmation, score in zip(starts, end_positions, confirmations, scores):
        rows.append({
            "candidate_start_index": int(indices[start_pos]),
            "candidate_end_index": int(indices[end_pos]),
            "candidate_start_position": int(start_pos),
            "candidate_end_position": int(end_pos),
            "candidate_confirmation_index": int(confirmation),
            "similarity": float(score),
            "pivot_type_sequence": " ".join(types[start_pos:end_pos + 1]),
        })
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def find_similar_structures(structure: pd.DataFrame, n_pivots: int = 8, top_k: int = 10, minimum_similarity: float = 0.0, config: SimilarityConfig | None = None, max_endpoint_index: int | None = None) -> pd.DataFrame:
    """Find historical matches for the latest confirmed structure."""
    if structure.empty:
        return pd.DataFrame(columns=["candidate_start_index", "candidate_end_index", "candidate_start_position", "candidate_end_position", "candidate_confirmation_index", "similarity", "pivot_type_sequence"])
    return find_similar_structures_at(structure, len(structure) - 1, n_pivots, top_k, minimum_similarity, config, max_endpoint_index)
