from __future__ import annotations

import pandas as pd


def validate_candles(df: pd.DataFrame, interval_ms: int | None = None) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return errors
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if df["open_time"].duplicated().any():
        errors.append("duplicate open timestamps")
    if (df["high"] < df["low"]).any():
        errors.append("high < low")
    if (df["open"] < df["low"]).any() or (df["open"] > df["high"]).any():
        errors.append("open outside high/low")
    if (df["close"] < df["low"]).any() or (df["close"] > df["high"]).any():
        errors.append("close outside high/low")
    if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
        errors.append("negative OHLCV value")
    if interval_ms and len(df) > 1:
        gaps = df["open_time"].sort_values().diff().dropna()
        if (gaps > interval_ms).any():
            errors.append("missing timestamp gaps detected")
    return errors
