from __future__ import annotations

import time

from data.binance import BinanceClient, INTERVAL_MS
from data.database import MarketDatabase
from data.validator import validate_candles
import pandas as pd


def download_timeframe(
    db: MarketDatabase,
    client: BinanceClient,
    symbol: str,
    timeframe: str,
    candles: int = 30_000,
) -> int:
    """Download the configured number of historical closed candles.

    On an empty timeframe, fetch up to ``candles`` closed candles ending at the
    latest completed candle. If history already exists, only fetch candles after
    the latest stored candle so routine updates remain incremental.
    """
    if timeframe not in INTERVAL_MS:
        raise ValueError(f"Unsupported Binance interval: {timeframe}")
    if candles <= 0:
        raise ValueError("candles must be greater than 0")

    now_ms = int(time.time() * 1000)
    step = INTERVAL_MS[timeframe]
    existing = db.last_open_time(symbol, timeframe)

    if existing is None:
        # Start far enough back to cover the requested number of completed
        # candles. We trim after normalization because the current candle is
        # intentionally excluded from the database.
        start = now_ms - (candles + 2) * step
    else:
        start = existing + step

    end = now_ms
    raw = client.download_range(symbol, timeframe, start, end)
    rows = client.normalize_klines(symbol, timeframe, raw, end)

    # Never persist the currently forming candle.
    rows = [r for r in rows if r[-1] == 1]

    # For an initial load, keep exactly the newest requested number of closed
    # candles. For an incremental update, keep every newly closed candle.
    if existing is None and len(rows) > candles:
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
