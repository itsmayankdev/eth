from __future__ import annotations

from collections.abc import Callable

import pandas as pd


SignalFn = Callable[[pd.DataFrame, int], object | None]


def walk_forward(df: pd.DataFrame, signal_fn: SignalFn, start: int) -> list[dict]:
    """Evaluate each timestamp using only rows at or before that timestamp.

    signal_fn receives a prefix ending at `i`; it must not inspect rows after i.
    Outcome evaluation is intentionally separate from signal generation.
    """
    results: list[dict] = []
    for i in range(max(start, 0), len(df)):
        prefix = df.iloc[: i + 1].copy()
        signal = signal_fn(prefix, i)
        results.append({"index": i, "open_time": df.iloc[i]["open_time"], "signal": signal})
    return results
