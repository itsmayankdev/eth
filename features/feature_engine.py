from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create dimensionless candle/price features without using future rows."""
    out = df.copy()
    eps = np.finfo(float).eps
    prev_close = out["close"].shift(1)
    out["return"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"] / prev_close)
    out["body"] = out["close"] - out["open"]
    out["body_abs"] = out["body"].abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["range"] = out["high"] - out["low"]
    out["body_range_ratio"] = out["body_abs"] / (out["range"] + eps)
    out["upper_wick_ratio"] = out["upper_wick"] / (out["range"] + eps)
    out["lower_wick_ratio"] = out["lower_wick"] / (out["range"] + eps)
    out["close_position"] = (out["close"] - out["low"]) / (out["range"] + eps)
    out["range_vs_mean"] = out["range"] / (out["range"].rolling(20, min_periods=5).mean() + eps)
    out["volume_change"] = out["volume"].pct_change()
    vol_mean = out["volume"].rolling(20, min_periods=5).mean()
    vol_std = out["volume"].rolling(20, min_periods=5).std()
    out["volume_z"] = (out["volume"] - vol_mean) / (vol_std + eps)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=5).mean()
    out["atr_pct"] = out["atr_14"] / (out["close"].abs() + eps)
    out["momentum_10"] = out["close"].pct_change(10)
    out["roc_20"] = out["close"].pct_change(20)
    out["volatility_20"] = out["log_return"].rolling(20, min_periods=5).std()
    out["slope_20"] = out["log_return"].rolling(20, min_periods=5).mean()
    out["acceleration"] = out["return"].diff()
    return out.replace([np.inf, -np.inf], np.nan)


def pattern_vector(df: pd.DataFrame, end_index: int, lookback: int) -> np.ndarray:
    start = end_index - lookback + 1
    if start < 0:
        raise ValueError("not enough rows for requested lookback")
    cols = ["return", "body_range_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_position", "range_vs_mean", "volume_z", "atr_pct", "momentum_10", "roc_20"]
    window = df.iloc[start:end_index + 1][cols].copy().ffill().fillna(0.0)
    # Standardize each feature inside the sequence so price regime is removed.
    arr = window.to_numpy(dtype=float)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    arr = (arr - mean) / (std + 1e-9)
    return arr.ravel()
