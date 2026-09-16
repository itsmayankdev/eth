from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from config import load_config
from data.database import MarketDatabase

app = FastAPI(title="ETH Market Structure Engine", version="0.1.0")


@app.get("/api/status")
def status() -> dict:
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    result = {"symbol": cfg.symbol, "timeframes": {}, "utc": datetime.now(timezone.utc).isoformat()}
    for tf in cfg.timeframes:
        result["timeframes"][tf] = db.last_open_time(cfg.symbol, tf)
    db.close()
    return result


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html><head><meta charset='utf-8'><title>ETH Market Structure Engine</title>
<style>body{font-family:system-ui;margin:32px;background:#0f1115;color:#eee} .card{padding:20px;border:1px solid #2a2e38;border-radius:12px;margin:12px 0} code{color:#9cdcfe}</style></head>
<body><h1>ETH Market Structure Engine</h1>
<div class='card'><strong>Research dashboard</strong><p>ETH/USDT structural analysis. Structural similarity and historical outcomes are intentionally separate.</p></div>
<div class='card'><h2>Database status</h2><pre id='status'>Loading…</pre></div>
<script>fetch('/api/status').then(r=>r.json()).then(x=>document.getElementById('status').textContent=JSON.stringify(x,null,2)).catch(e=>document.getElementById('status').textContent=e)</script>
</body></html>"""
