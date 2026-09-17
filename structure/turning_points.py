from __future__ import annotations

import pandas as pd


def zigzag_turns(df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Detect confirmed ZigZag turns and record when each turn became knowable.

    ``index`` is the pivot candle where the extreme occurred.  ``confirmation_index``
    is the later candle that confirmed the reversal by threshold_pct.  Keeping both
    is important for causal backtests: the pivot price itself is not knowable at the
    pivot candle until the reversal has subsequently been confirmed.
    """
    columns = ["index", "confirmation_index", "direction", "price"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    threshold = threshold_pct / 100.0
    closes = df["close"].to_numpy(float)
    events: list[tuple[int, int, str, float]] = []
    pivot_idx = 0
    pivot_price = closes[0]
    trend: str | None = None

    for i in range(1, len(closes)):
        price = closes[i]
        move = price / pivot_price - 1.0

        if trend is None:
            if move >= threshold:
                trend = "UP"
                pivot_idx, pivot_price = i, price
            elif move <= -threshold:
                trend = "DOWN"
                pivot_idx, pivot_price = i, price
            continue

        if trend == "UP":
            if price > pivot_price:
                pivot_idx, pivot_price = i, price
            elif price / pivot_price - 1.0 <= -threshold:
                events.append((pivot_idx, i, "DOWN", pivot_price))
                trend = "DOWN"
                pivot_idx, pivot_price = i, price
        else:
            if price < pivot_price:
                pivot_idx, pivot_price = i, price
            elif price / pivot_price - 1.0 >= threshold:
                events.append((pivot_idx, i, "UP", pivot_price))
                trend = "UP"
                pivot_idx, pivot_price = i, price

    return pd.DataFrame(events, columns=columns)
