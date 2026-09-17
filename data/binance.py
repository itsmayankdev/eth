from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

REST_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


class BinanceClient:
    def __init__(self, base_url: str = REST_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None, limit: int = 1000) -> list[list[Any]]:
        # Binance rejects negative timestamps. For very long requests (notably
        # 1d/4h), the requested history can pre-date the Unix epoch; clamping to
        # zero lets the API return the earliest available candles for the symbol.
        start_ms = max(0, int(start_ms))
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": min(limit, 1000)}
        if end_ms is not None:
            params["endTime"] = end_ms
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()

    def download_range(self, symbol: str, interval: str, start_ms: int, end_ms: int, max_candles: int | None = None) -> list[list[Any]]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported Binance interval: {interval}")
        rows: list[list[Any]] = []
        cursor = max(0, int(start_ms))
        step = INTERVAL_MS[interval]
        target = max_candles if max_candles is not None else float("inf")
        while cursor <= end_ms and len(rows) < target:
            remaining = int(target - len(rows)) if target != float("inf") else 1000
            limit = min(1000, max(1, remaining))
            batch = self.fetch_klines(symbol, interval, cursor, end_ms, limit=limit)
            if not batch:
                break
            rows.extend(batch[:remaining])
            last = int(batch[-1][0])
            next_cursor = last + step
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
            time.sleep(0.15)
        return rows[:max_candles] if max_candles is not None else rows

    @staticmethod
    def normalize_klines(symbol: str, interval: str, raw: list[list[Any]], now_ms: int | None = None) -> list[tuple]:
        now_ms = now_ms or int(time.time() * 1000)
        step = INTERVAL_MS[interval]
        result = []
        for r in raw:
            open_time = int(r[0])
            result.append((symbol, interval, open_time, int(r[6]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7]), int(r[8]), int(open_time + step <= now_ms)))
        return result


async def websocket_closed_candles(symbol: str, interval: str, websocket_base: str, reconnect_seconds: int = 5) -> AsyncIterator[dict[str, Any]]:
    stream = f"{symbol.lower()}@kline_{interval}"
    url = f"{websocket_base.rstrip('/')}/{stream}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                async for message in ws:
                    import json
                    payload = json.loads(message)
                    kline = payload.get("k", {})
                    if kline.get("x"):
                        yield payload
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException):
            await asyncio.sleep(reconnect_seconds)
