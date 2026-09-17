"""Small deterministic checks for the candle similarity engine."""
import pandas as pd

from candle_similarity import candle_features, candle_sequence_similarity


def _df(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_identical_sequences_score_one():
    df = _df([
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 101.5),
        (101.5, 103, 100, 102.5),
    ])
    f = candle_features(df)
    assert candle_sequence_similarity(f, f) == 1.0


def test_different_sequences_are_not_more_similar_than_identical():
    a = _df([
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
        (103, 105, 102, 104),
    ])
    b = _df([
        (100, 101, 99, 99.5),
        (99.5, 101, 98, 100.5),
        (100.5, 101, 96, 97),
        (97, 100, 95, 99),
    ])
    fa, fb = candle_features(a), candle_features(b)
    assert candle_sequence_similarity(fa, fa) >= candle_sequence_similarity(fa, fb)
