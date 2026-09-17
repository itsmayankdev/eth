from __future__ import annotations

import pandas as pd

from structure.classification import classify_swings
from structure.turning_points import zigzag_turns


STRUCTURE_COLUMNS = [
    "index",
    "direction",
    "price",
    "pivot_type",
    "previous_same_type_price",
    "swing_pct",
    "bars_since_prev",
    "leg_pct",
    "normalized_price",
]


def build_structure(candles: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Build a normalized, multi-scale market-structure sequence.

    The representation keeps the confirmed ZigZag pivots and adds:
    - HH/HL/LH/LL classification;
    - bars between consecutive pivots;
    - percentage change from the previous pivot of any type;
    - price normalized to the first pivot in the sequence.

    No future candles are used beyond the confirmation already required by
    the ZigZag detector, so this representation is suitable for walk-forward
    research once the final, unconfirmed pivot is excluded by the detector.
    """
    turns = zigzag_turns(candles, threshold_pct)
    out = classify_swings(turns)
    if out.empty:
        return pd.DataFrame(columns=STRUCTURE_COLUMNS)

    out = out.copy()
    out["index"] = out["index"].astype(int)
    out["price"] = out["price"].astype(float)
    out["bars_since_prev"] = out["index"].diff()
    out["leg_pct"] = out["price"].pct_change() * 100.0
    first_price = float(out.iloc[0]["price"])
    out["normalized_price"] = out["price"] / first_price

    return out[STRUCTURE_COLUMNS]


def structure_signature(structure: pd.DataFrame, n_pivots: int = 8) -> list[dict]:
    """Return the latest n-pivot structure as JSON-friendly records."""
    if structure.empty or n_pivots <= 0:
        return []
    cols = ["pivot_type", "direction", "swing_pct", "bars_since_prev", "leg_pct"]
    tail = structure.tail(n_pivots)[cols].copy()
    return tail.where(pd.notna(tail), None).to_dict(orient="records")
