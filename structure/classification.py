from __future__ import annotations

import pandas as pd


def classify_swings(turns: pd.DataFrame) -> pd.DataFrame:
    """Classify confirmed ZigZag pivots as HH, HL, LH, or LL.

    The ZigZag output labels each pivot by the direction of the reversal that
    confirmed it. ``DOWN`` marks a swing high and ``UP`` marks a swing low.
    The confirmation index is preserved so causal evaluation knows when the
    pivot became observable.
    """
    columns = [
        "index",
        "confirmation_index",
        "direction",
        "price",
        "pivot_type",
        "previous_same_type_price",
        "swing_pct",
    ]
    if turns.empty:
        return pd.DataFrame(columns=columns)

    required = {"index", "confirmation_index", "direction", "price"}
    missing = sorted(required - set(turns.columns))
    if missing:
        raise ValueError(f"missing turning-point columns: {missing}")

    out = turns.copy().reset_index(drop=True)
    out["pivot_type"] = pd.NA
    out["previous_same_type_price"] = pd.NA
    out["swing_pct"] = pd.NA

    last_high: float | None = None
    last_low: float | None = None

    for i, row in out.iterrows():
        price = float(row["price"])
        if row["direction"] == "DOWN":
            previous = last_high
            out.at[i, "previous_same_type_price"] = previous
            if previous is not None:
                out.at[i, "pivot_type"] = "HH" if price > previous else "LH"
                out.at[i, "swing_pct"] = (price / previous - 1.0) * 100.0
            last_high = price
        elif row["direction"] == "UP":
            previous = last_low
            out.at[i, "previous_same_type_price"] = previous
            if previous is not None:
                out.at[i, "pivot_type"] = "HL" if price > previous else "LL"
                out.at[i, "swing_pct"] = (price / previous - 1.0) * 100.0
            last_low = price

    out["index"] = out["index"].astype(int)
    out["confirmation_index"] = out["confirmation_index"].astype(int)
    return out[columns]
