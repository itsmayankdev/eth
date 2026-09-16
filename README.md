# ETH Market Structure Engine

Research-focused Ethereum (ETH/USDT) market-structure and historical-pattern matching platform.

## Scope

- ETH/USDT only
- Binance market data
- 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
- Local SQLite persistence
- Incremental historical updates
- Mathematical candle/price features
- Turning-point detection
- Normalized multi-candle pattern representations
- Historical similarity search
- Outcome/MFE/MAE analysis
- Walk-forward-safe analysis primitives
- Live WebSocket scanner foundation
- FastAPI + Plotly dashboard foundation

The design follows the project specification: structural similarity is kept separate from historical outcome statistics and no future candles are used to construct a live pattern.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python main.py init
python main.py download --days 365
python main.py update
python main.py scan
uvicorn dashboard.app:app --reload
```

Data is stored in `data/eth_market.db`. Generated reports go under `reports/`.

## Configuration

Edit `config.yaml` for symbol, timeframes, history length, lookbacks, similarity threshold, and outcome horizons.

## Safety / research note

This is a research and statistical analysis system, not an automated trading or investment-advice system. Historical similarity does not imply future profitability.
