from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.binance import BinanceClient, INTERVAL_MS
from data.database import MarketDatabase
from data.validator import validate_candles
import pandas as pd


def download_timeframe(db: MarketDatabase, client: BinanceClient, symbol: str, timeframe: str, days: int = 365) -> int:
    now = datetime.now(timezone.utc)
    existing = db.last_open_time(symbol, timeframe)
    start = existing + INTERVAL_MS[timeframe] if existing is not None else int((now - timedelta(days=days)).timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    raw = client.download_range(symbol, timeframe, start, end)
    rows = client.normalize_klines(symbol, timeframe, raw, end)
    # Do not persist the currently forming candle.
    rows = [r for r in rows if r[-1] == 1]
    if rows:
        check = pd.DataFrame(rows, columns=["symbol","timeframe","open_time","close_time","open","high","low","close","volume","quote_volume","trades","is_closed"])
        errors = validate_candles(check, INTERVAL_MS[timeframe])
        if errors:
            raise ValueError(f"{timeframe} validation failed: {errors}")
        return db.upsert_candles(rows)
    return 0
