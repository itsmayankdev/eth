from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutcomeConfig:
    horizons: tuple[int, ...] = (5, 10, 20, 50)
    target_pct: float = 1.0
    stop_pct: float = 0.5

    def __post_init__(self) -> None:
        if not self.horizons or any(int(h) <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.target_pct <= 0 or self.stop_pct <= 0:
            raise ValueError("target_pct and stop_pct must be greater than 0")


def _forward_window(candles: pd.DataFrame, endpoint_index: int, horizon: int) -> pd.DataFrame:
    """Return candles strictly after the signal/entry candle."""
    if candles.empty or horizon <= 0:
        return candles.iloc[0:0]
    if "index" in candles.columns:
        return candles[candles["index"] > endpoint_index].head(horizon)
    return candles.iloc[endpoint_index + 1 : endpoint_index + 1 + horizon]


def measure_match_outcome(
    candles: pd.DataFrame,
    endpoint_index: int,
    entry_price: float,
    horizons: tuple[int, ...] = (5, 10, 20, 50),
    target_pct: float = 1.0,
    stop_pct: float = 0.5,
) -> dict:
    """Measure returns, MFE/MAE and target/stop outcomes after a causal entry.

    ``endpoint_index`` is the candle where the signal is available. Forward
    candles begin strictly after it. The caller should use the ZigZag
    ``confirmation_index`` rather than the earlier pivot index.
    """
    result: dict = {"endpoint_index": int(endpoint_index), "entry_price": float(entry_price)}
    target = float(entry_price) * (1.0 + target_pct / 100.0)
    stop = float(entry_price) * (1.0 - stop_pct / 100.0)

    max_horizon = max(int(h) for h in horizons)
    future = _forward_window(candles, endpoint_index, max_horizon)
    if future.empty:
        for h in horizons:
            result[f"return_{h}"] = np.nan
            result[f"mfe_{h}"] = np.nan
            result[f"mae_{h}"] = np.nan
            result[f"target_first_{h}"] = "no_data"
        return result

    highs = pd.to_numeric(future["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(future["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(future["close"], errors="coerce").to_numpy(float)
    up = (highs / entry_price - 1.0) * 100.0
    down = (lows / entry_price - 1.0) * 100.0

    for h in horizons:
        h = int(h)
        hh = highs[:h]
        ll = lows[:h]
        cc = closes[:h]
        if len(cc) == 0 or not np.isfinite(cc[-1]):
            result[f"return_{h}"] = np.nan
            result[f"mfe_{h}"] = np.nan
            result[f"mae_{h}"] = np.nan
            result[f"target_first_{h}"] = "no_data"
            continue

        result[f"return_{h}"] = float((cc[-1] / entry_price - 1.0) * 100.0)
        result[f"mfe_{h}"] = float(np.nanmax(up[: len(hh)]))
        result[f"mae_{h}"] = float(np.nanmin(down[: len(ll)]))

        target_hits = np.flatnonzero(hh >= target)
        stop_hits = np.flatnonzero(ll <= stop)
        target_i = int(target_hits[0]) if len(target_hits) else None
        stop_i = int(stop_hits[0]) if len(stop_hits) else None

        if target_i is None and stop_i is None:
            outcome = "neither"
        elif target_i is not None and stop_i is None:
            outcome = "target"
        elif stop_i is not None and target_i is None:
            outcome = "stop"
        elif target_i < stop_i:
            outcome = "target"
        elif stop_i < target_i:
            outcome = "stop"
        else:
            outcome = "ambiguous"
        result[f"target_first_{h}"] = outcome

    return result


def evaluate_matches(
    matches: pd.DataFrame,
    candles: pd.DataFrame,
    structure: pd.DataFrame,
    config: OutcomeConfig | None = None,
) -> pd.DataFrame:
    """Attach strictly-forward outcomes using each match's confirmation candle.

    Entry is the close of the confirmation candle. This avoids assuming that a
    historical pivot price was tradable before the reversal had been confirmed.
    """
    cfg = config or OutcomeConfig()
    if matches.empty:
        return matches.copy()

    rows: list[dict] = []
    closes = pd.to_numeric(candles["close"], errors="coerce").to_numpy(float)
    for _, match in matches.iterrows():
        end_pos = int(match["candidate_end_position"])
        pivot = structure.iloc[end_pos]
        confirmation_index = int(match.get("candidate_confirmation_index", pivot["confirmation_index"]))
        if confirmation_index < 0 or confirmation_index >= len(closes) or not np.isfinite(closes[confirmation_index]):
            continue
        entry_price = float(closes[confirmation_index])
        outcome = measure_match_outcome(
            candles,
            confirmation_index,
            entry_price,
            cfg.horizons,
            cfg.target_pct,
            cfg.stop_pct,
        )
        row = match.to_dict()
        row.update(outcome)
        row["entry_index"] = confirmation_index
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_outcomes(outcomes: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Aggregate target/stop results by horizon without inventing outcomes."""
    rows: list[dict] = []
    for h in horizons:
        col = f"target_first_{int(h)}"
        if col not in outcomes:
            continue
        values = outcomes[col].dropna().astype(str)
        decisive = values[values.isin(["target", "stop"])]
        n = len(values)
        rows.append(
            {
                "horizon": int(h),
                "samples": n,
                "target": int((values == "target").sum()),
                "stop": int((values == "stop").sum()),
                "neither": int((values == "neither").sum()),
                "ambiguous": int((values == "ambiguous").sum()),
                "target_rate_all": float((values == "target").mean()) if n else np.nan,
                "target_rate_decisive": float((decisive == "target").mean()) if len(decisive) else np.nan,
            }
        )
    return pd.DataFrame(rows)
