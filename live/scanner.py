from __future__ import annotations

import asyncio
import json

from config import Config
from data.binance import INTERVAL_MS, websocket_closed_candles
from data.database import MarketDatabase


class LiveScanner:
    def __init__(self, config: Config, db: MarketDatabase) -> None:
        self.config = config
        self.db = db

    async def run_timeframe(self, timeframe: str) -> None:
        async for payload in websocket_closed_candles(
            self.config.symbol,
            timeframe,
            self.config.live.websocket_base,
            self.config.live.reconnect_seconds,
        ):
            k = payload["k"]
            row = (
                self.config.symbol,
                timeframe,
                int(k["t"]),
                int(k["T"]),
                float(k["o"]),
                float(k["h"]),
                float(k["l"]),
                float(k["c"]),
                float(k["v"]),
                float(k["q"]),
                int(k["n"]),
                1,
            )
            self.db.upsert_candles([row])

    async def run(self) -> None:
        await asyncio.gather(*(self.run_timeframe(tf) for tf in self.config.timeframes))


def run_live(config: Config, db: MarketDatabase) -> None:
    asyncio.run(LiveScanner(config, db).run())
