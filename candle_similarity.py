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
    distance = np.sqrt(np.average((a - b) ** 2, axis=(0, 1), weights=None))
    # Feature-aware weighted RMSE; weights are applied across columns.
    per_feature = np.sqrt(np.mean((a - b) ** 2, axis=0))
    distance = float(np.sum(per_feature * w))
    return float(np.exp(-distance))


def rank_candle_sequences(
    features: pd.DataFrame,
    current_start: int,
    current_end: int,
    candidate_ends: Iterable[int],
    config: CandleSimilarityConfig | None = None,
) -> pd.DataFrame:
    """Rank historical sequences ending before the current sequence."""
    config = config or CandleSimilarityConfig()
    length = current_end - current_start + 1
    current = features.iloc[current_start:current_end + 1]
    rows = []
    for end in candidate_ends:
        start = end - length + 1
        if start < 0 or end >= current_start:
            continue
        candidate = features.iloc[start:end + 1]
        score = candle_sequence_similarity(current, candidate, config)
        rows.append({"candidate_start": start, "candidate_end": end, "similarity": score})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


__all__ = [
    "FEATURES", "CandleSimilarityConfig", "candle_features",
    "candle_sequence_similarity", "rank_candle_sequences", "sequence_matrix",
]
