from __future__ import annotations

import math

import numpy as np
import pandas as pd


def outcome_for_event(df: pd.DataFrame, event_index: int, horizon: int) -> dict[str, float | None]:
    if event_index < 0 or event_index >= len(df):
        raise IndexError(event_index)
    end = min(len(df) - 1, event_index + horizon)
    entry = float(df.iloc[event_index]["close"])
    future = df.iloc[event_index + 1:end + 1]
    if future.empty:
        return {"forward_return": None, "mfe": None, "mae": None, "max_up": None, "max_down": None}
    returns_close = future["close"] / entry - 1.0
    up = future["high"] / entry - 1.0
    down = future["low"] / entry - 1.0
    return {
        "forward_return": float(returns_close.iloc[-1]),
        "mfe": float(up.max()),
        "mae": float(down.min()),
        "max_up": float(up.max()),
        "max_down": float(down.min()),
    }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return (math.nan, math.nan)
    # Normal approximation z; avoids scipy dependency in the storage/report layer.
    z_table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_table.get(round(confidence, 2), 1.959963984540054)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_returns(returns: list[float] | np.ndarray) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"sample_size": 0}
    return {
        "sample_size": int(len(values)),
        "average_return": float(values.mean()),
        "median_return": float(np.median(values)),
        "max_return": float(values.max()),
        "min_return": float(values.min()),
        "up_count": int((values > 0).sum()),
        "down_count": int((values < 0).sum()),
        "neutral_count": int((values == 0).sum()),
    }
