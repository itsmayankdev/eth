"""Candle-behaviour similarity research engine.

This module compares raw OHLC candle sequences independently of the existing
HH/HL/LH/LL structural matcher. It is descriptive research tooling only:
no entries, targets, stops, outcomes, or predictions are calculated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FEATURES = (
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_ratio",
    "close_location",
    "direction",
    "body_change",
    "range_change",
    "alternation",
)


@dataclass(frozen=True)
class CandleSimilarityConfig:
    body_weight: float = 0.18
    upper_wick_weight: float = 0.12
    lower_wick_weight: float = 0.12
    range_weight: float = 0.16
    close_location_weight: float = 0.12
    direction_weight: float = 0.10
    body_change_weight: float = 0.08
    range_change_weight: float = 0.08
    alternation_weight: float = 0.04

    @property
    def weights(self) -> np.ndarray:
        return np.array([
            self.body_weight, self.upper_wick_weight, self.lower_wick_weight,
            self.range_weight, self.close_location_weight, self.direction_weight,
            self.body_change_weight, self.range_change_weight, self.alternation_weight,
        ], dtype=float)


def candle_features(df: pd.DataFrame, median_window: int = 20) -> pd.DataFrame:
    """Return normalized candle-anatomy and local-behaviour features."""
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    x = df.reset_index(drop=True).copy()
    eps = 1e-12
    rng = (x["high"] - x["low"]).clip(lower=eps)
    body = (x["close"] - x["open"]).abs()
    upper = x["high"] - x[["open", "close"]].max(axis=1)
    lower = x[["open", "close"]].min(axis=1) - x["low"]
    median_range = rng.rolling(median_window, min_periods=1).median().clip(lower=eps)

    out = pd.DataFrame(index=x.index)
    out["body_pct"] = (body / rng).clip(0, 1)
    out["upper_wick_pct"] = (upper / rng).clip(0, 1)
    out["lower_wick_pct"] = (lower / rng).clip(0, 1)
    out["range_ratio"] = (rng / median_range).clip(0, 5) / 5.0
    out["close_location"] = ((x["close"] - x["low"]) / rng).clip(0, 1)
    out["direction"] = np.sign(x["close"] - x["open"])

    body_pct = out["body_pct"]
    range_ratio = out["range_ratio"]
    out["body_change"] = body_pct.diff().fillna(0).clip(-1, 1)
    out["range_change"] = range_ratio.diff().fillna(0).clip(-1, 1)
    prev_dir = out["direction"].shift(1)
    out["alternation"] = ((out["direction"] * prev_dir) < 0).astype(float)
    out.loc[prev_dir.isna(), "alternation"] = 0.0
    return out


def _standardize_sequence(a: np.ndarray) -> np.ndarray:
    """Normalize a sequence feature column without destroying its shape."""
    med = np.nanmedian(a, axis=0)
    scale = np.nanmedian(np.abs(a - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (a - med) / scale


def sequence_matrix(features: pd.DataFrame, start: int, end: int) -> np.ndarray:
    if start < 0 or end >= len(features) or start > end:
        raise IndexError("Invalid candle sequence bounds")
    return features.loc[start:end, list(FEATURES)].to_numpy(dtype=float)


def candle_sequence_similarity(
    current: pd.DataFrame,
    candidate: pd.DataFrame,
    config: CandleSimilarityConfig | None = None,
) -> float:
    """Compare two equal-length candle sequences on anatomy/behaviour.

    Returns 0..1, where 1 means identical normalized feature sequences.
    """
    config = config or CandleSimilarityConfig()
    if len(current) != len(candidate) or len(current) == 0:
        raise ValueError("Sequences must have equal, non-zero length")

    a = _standardize_sequence(current[list(FEATURES)].to_numpy(dtype=float))
    b = _standardize_sequence(candidate[list(FEATURES)].to_numpy(dtype=float))
    w = config.weights
    w = w / w.sum()
    per_feature = np.sqrt(np.mean((a - b) ** 2, axis=0))
    distance = float(np.sum(per_feature * w))
    return float(np.exp(-distance))


def rank_candle_sequences(
    features: pd.DataFrame,
    current_start: int,
    current_end: int,
    candidate_ends: Iterable[int],
    config: CandleSimilarityConfig | None = None,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    """Rank historical sequences with vectorized, chunked similarity.

    This preserves the same per-sequence robust standardization and feature
    weighting as :func:`candle_sequence_similarity`, but compares many
    historical windows at once instead of calling the scalar function in a
    Python loop. Candidate windows ending at or after ``current_start`` are
    excluded so the historical sequence cannot overlap the current sequence.
    """
    config = config or CandleSimilarityConfig()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    length = current_end - current_start + 1
    if length <= 0:
        raise ValueError("Invalid current sequence bounds")
    if current_start < length - 1:
        raise ValueError("Not enough history for the requested sequence length")

    matrix = features.loc[:, list(FEATURES)].to_numpy(dtype=float)
    current = matrix[current_start:current_end + 1]
    current_std = _standardize_sequence(current)
    weights = config.weights
    weights = weights / weights.sum()

    ends = np.asarray(list(candidate_ends), dtype=int)
    ends = ends[(ends >= length - 1) & (ends < current_start)]
    if ends.size == 0:
        return pd.DataFrame(columns=["candidate_start", "candidate_end", "similarity"])
    ends = np.unique(ends)

    # One rolling-window view is created once. Chunking keeps temporary
    # standardized arrays bounded while retaining vectorized NumPy work.
    windows = np.lib.stride_tricks.sliding_window_view(matrix, length, axis=0)
    windows = np.moveaxis(windows, -1, 1)  # (n_windows, length, n_features)

    scored = []
    for start_i in range(0, len(ends), chunk_size):
        chunk_ends = ends[start_i:start_i + chunk_size]
        chunk_starts = chunk_ends - length + 1
        batch = windows[chunk_starts]

        med = np.nanmedian(batch, axis=1, keepdims=True)
        scale = np.nanmedian(np.abs(batch - med), axis=1, keepdims=True) * 1.4826
        scale = np.where(scale < 1e-6, 1.0, scale)
        batch_std = (batch - med) / scale

        per_feature = np.sqrt(np.mean((batch_std - current_std[None, :, :]) ** 2, axis=1))
        distance = np.sum(per_feature * weights[None, :], axis=1)
        similarity = np.exp(-distance)

        scored.append(np.column_stack((chunk_starts, chunk_ends, similarity)))

    result = np.vstack(scored)
    out = pd.DataFrame(result, columns=["candidate_start", "candidate_end", "similarity"])
    out["candidate_start"] = out["candidate_start"].astype(int)
    out["candidate_end"] = out["candidate_end"].astype(int)
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


__all__ = [
    "FEATURES", "CandleSimilarityConfig", "candle_features",
    "candle_sequence_similarity", "rank_candle_sequences", "sequence_matrix",
]
