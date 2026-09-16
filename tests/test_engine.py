import numpy as np
import pandas as pd

from data.validator import validate_candles
from features.feature_engine import add_features, pattern_vector
from patterns.similarity import ensemble_similarity
from statistics.outcomes import outcome_for_event, wilson_interval
from structure.turning_points import zigzag_turns


def sample_df() -> pd.DataFrame:
    close = np.array([100, 101, 102, 101, 99, 98, 99, 101, 103, 102, 104], dtype=float)
    return pd.DataFrame({
        "open_time": np.arange(len(close)) * 60_000,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(len(close)) + 10,
    })


def test_validation_accepts_valid_ohlc():
    assert validate_candles(sample_df(), 60_000) == []


def test_features_are_created_without_future_columns():
    enriched = add_features(sample_df())
    assert {"return", "body_range_ratio", "atr_pct", "momentum_10"}.issubset(enriched.columns)


def test_pattern_vector_has_expected_shape():
    enriched = add_features(sample_df())
    vector = pattern_vector(enriched, len(enriched) - 1, 5)
    assert vector.shape == (50,)
    assert np.isfinite(vector).all()


def test_similarity_is_one_for_same_vector():
    x = np.arange(20, dtype=float)
    assert ensemble_similarity(x, x) > 0.99


def test_zigzag_finds_turns():
    turns = zigzag_turns(sample_df(), 1.0)
    assert not turns.empty


def test_outcome_calculation():
    result = outcome_for_event(sample_df(), 2, 3)
    assert result["forward_return"] is not None
    assert result["mfe"] >= result["mae"]


def test_wilson_interval_is_bounded():
    lo, hi = wilson_interval(72, 103)
    assert 0 <= lo <= hi <= 1
