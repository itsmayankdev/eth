from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    close_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    trades INTEGER,
    is_closed INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (symbol, timeframe, open_time)
);
CREATE INDEX IF NOT EXISTS idx_candles_tf_time ON candles(timeframe, open_time);

CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    feature_json TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, open_time),
    FOREIGN KEY (symbol, timeframe, open_time)
      REFERENCES candles(symbol, timeframe, open_time) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turning_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    direction TEXT NOT NULL,
    threshold_pct REAL NOT NULL,
    price REAL NOT NULL,
    UNIQUE(symbol, timeframe, open_time, threshold_pct)
);

CREATE TABLE IF NOT EXISTS pattern_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    direction TEXT,
    lookback INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    UNIQUE(symbol, timeframe, open_time, lookback)
);

CREATE TABLE IF NOT EXISTS outcomes (
    event_id INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    forward_return REAL,
    mfe REAL,
    mae REAL,
    max_up REAL,
    max_down REAL,
    PRIMARY KEY(event_id, horizon),
    FOREIGN KEY(event_id) REFERENCES pattern_events(id) ON DELETE CASCADE
);
"""


class MarketDatabase:
    def __init__(self, path: str = "data/eth_market.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def last_open_time(self, symbol: str, timeframe: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(open_time) FROM candles WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        ).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def upsert_candles(self, rows: Iterable[tuple]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        self.conn.executemany(
            """INSERT INTO candles
            (symbol,timeframe,open_time,close_time,open,high,low,close,volume,quote_volume,trades,is_closed)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,timeframe,open_time) DO UPDATE SET
              close_time=excluded.close_time, open=excluded.open, high=excluded.high,
              low=excluded.low, close=excluded.close, volume=excluded.volume,
              quote_volume=excluded.quote_volume, trades=excluded.trades,
              is_closed=excluded.is_closed""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_candles(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM candles WHERE symbol=? AND timeframe=? ORDER BY open_time"
        params: list = [symbol, timeframe]
        if limit:
            sql = "SELECT * FROM (" + sql + " DESC LIMIT ?) ORDER BY open_time"
            params.append(limit)
        return pd.read_sql_query(sql, self.conn, params=params)
