from __future__ import annotations

import time

from data.binance import BinanceClient, INTERVAL_MS
from data.database import MarketDatabase
from data.validator import validate_candles
import pandas as pd


def _keep_latest_contiguous(rows: list[tuple], step: int) -> list[tuple]:
    """Keep the newest contiguous candle segment when older history has a gap."""
    if len(rows) < 2:
        return rows
    ordered = sorted(rows, key=lambda r: r[2])
    timestamps = [int(r[2]) for r in ordered]
    last_gap = -1
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > step:
            last_gap = i
    if last_gap >= 0:
        return ordered[last_gap:]
    return ordered


def download_timeframe(
    db: MarketDatabase,
    client: BinanceClient,
    symbol: str,
    timeframe: str,
    candles: int = 30_000,
) -> int:
    """Download up to the configured number of latest closed candles.

    The initial request is aligned to the timeframe boundary. If the requested
    count reaches further back than the symbol's available history, Binance
    simply returns the available history. If an older portion contains a gap,
    only the newest contiguous segment is retained so structural calculations
    are not built across missing candles.
    """
    if timeframe not in INTERVAL_MS:
        raise ValueError(f"Unsupported Binance interval: {timeframe}")
    if candles <= 0:
        raise ValueError("candles must be greater than 0")

    now_ms = int(time.time() * 1000)
    step = INTERVAL_MS[timeframe]
    existing = db.last_open_time(symbol, timeframe)

    if existing is None:
        # Align to the Binance candle boundary. Exclude the currently forming
        # candle by ending at the previous boundary.
        latest_closed_open = (now_ms // step) * step - step
        start = latest_closed_open - (candles - 1) * step
        end = latest_closed_open + step - 1
    else:
        start = existing + step
        end = now_ms

    raw = client.download_range(
        symbol,
        timeframe,
        start,
        end,
        max_candles=candles + 2 if existing is None else None,
    )
    rows = client.normalize_klines(symbol, timeframe, raw, now_ms)

    # Never persist the currently forming candle.
    rows = [r for r in rows if r[-1] == 1]

    if existing is None:
        rows = _keep_latest_contiguous(rows, step)
        if len(rows) > candles:
            rows = rows[-candles:]

    if rows:
        check = pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "timeframe",
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "is_closed",
            ],
        )
        errors = validate_candles(check, step)
        if errors:
            raise ValueError(f"{timeframe} validation failed: {errors}")
        return db.upsert_candles(rows)
    return 0
