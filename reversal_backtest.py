from __future__ import annotations
"""Historical ETH reversal backtest and visual comparison engine.

This is the backend validation layer for the candle-research project.
It finds confirmed market reversals with a ZigZag threshold, highlights the
pivot and confirmation candles, measures what happened afterwards, and finds
earlier reversal episodes with similar 20-candle behaviour.

It is research/backtesting infrastructure: no live signals or trade execution.
Forward measurements are descriptive signed returns, MFE/MAE-style movement,
and persistence of the reversal direction.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from price_level_interaction_v2 import load_market
from structure.turning_points import zigzag_turns
from candle_similarity import candle_features

TIMEFRAMES = ("1m","3m","5m","15m","30m","1h","4h","1d")
THRESHOLDS = (0.25, 0.50, 1.00, 2.00)
WINDOW = 20
HORIZONS = (5, 10, 20, 50)
TOP_MATCHES = 10
MIN_GAP = 5

def seq_vector(feat: pd.DataFrame, start: int, end: int) -> np.ndarray:
    a = feat.iloc[start:end+1].to_numpy(float)
    # 9 candle-behaviour features x 20 candles, with per-sequence robust scaling.
    med = np.nanmedian(a, axis=0)
    scale = np.nanmedian(np.abs(a-med), axis=0)*1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((a-med)/scale).reshape(-1)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size: return 0.0
    d = float(np.sqrt(np.mean((a-b)**2)))
    return float(np.exp(-d))

def signed_forward(df: pd.DataFrame, idx: int, direction: str) -> dict:
    p = float(df.iloc[idx].close)
    out = {"confirmation_close": p}
    sign = 1.0 if direction == "UP" else -1.0
    for h in HORIZONS:
        f = df.iloc[idx+1:idx+1+h]
        if f.empty:
            out[f"signed_return_{h}"] = np.nan
            out[f"max_favorable_{h}"] = np.nan
            out[f"max_adverse_{h}"] = np.nan
            continue
        c = f.close.to_numpy(float)
        hi = f.high.to_numpy(float); lo = f.low.to_numpy(float)
        ret = (c[-1]/p-1.0)*100.0*sign
        favorable = ((hi/p-1.0)*100.0 if direction=="UP" else (p/lo-1.0)*100.0)
        adverse = ((lo/p-1.0)*100.0 if direction=="UP" else (p/hi-1.0)*100.0)
        out[f"signed_return_{h}"] = float(ret)
        out[f"max_favorable_{h}"] = float(np.nanmax(favorable))
        out[f"max_adverse_{h}"] = float(np.nanmin(adverse))
    return out

def event_rows(df: pd.DataFrame, turns: pd.DataFrame, feat: pd.DataFrame, threshold: float) -> list[dict]:
    rows=[]
    for ridx,t in turns.iterrows():
        pivot=int(t["index"]); confirm=int(t["confirmation_index"])
        if confirm < WINDOW-1 or confirm+1 >= len(df): continue
        direction=str(t["direction"])
        row={
            "event_id": f"R{ridx+1:05d}",
            "pivot_index": pivot,
            "confirmation_index": confirm,
            "direction": direction,
            "pivot_price": float(t["price"]),
            "pivot_time": int(df.iloc[pivot].open_time),
            "confirmation_time": int(df.iloc[confirm].open_time),
            "confirmation_close": float(df.iloc[confirm].close),
            "confirmation_delay_candles": confirm-pivot,
            "pre_window_start": confirm-WINDOW+1,
            "pre_window_end": confirm,
        }
        row.update(signed_forward(df,confirm,direction))
        rows.append(row)
    return rows

def find_matches(events: list[dict], feat: pd.DataFrame, top_k: int=TOP_MATCHES) -> pd.DataFrame:
    if not events: return pd.DataFrame()
    vectors={e["event_id"]:seq_vector(feat,e["pre_window_start"],e["pre_window_end"]) for e in events}
    rows=[]
    for i,e in enumerate(events):
        candidates=[]
        for j,c in enumerate(events[:i]):
            # Same reversal direction is the primary comparison; opposite direction
            # is retained separately so the backend can show both.
            if e["direction"] != c["direction"]: continue
            if abs(e["confirmation_index"]-c["confirmation_index"]) < MIN_GAP: continue
            s=similarity(vectors[e["event_id"]],vectors[c["event_id"]])
            candidates.append((s,c))
        candidates.sort(key=lambda x:x[0],reverse=True)
        for rank,(s,c) in enumerate(candidates[:top_k],1):
            rows.append({
                "event_id":e["event_id"],"match_rank":rank,
                "reversal_direction":e["direction"],
                "event_confirmation_time":e["confirmation_time"],
                "match_event_id":c["event_id"],
                "match_confirmation_time":c["confirmation_time"],
                "similarity":s,
                "match_pivot_price":c["pivot_price"],
                "match_confirmation_close":c["confirmation_close"],
                "match_signed_return_5":c["signed_return_5"],
                "match_signed_return_10":c["signed_return_10"],
                "match_signed_return_20":c["signed_return_20"],
                "match_signed_return_50":c["signed_return_50"],
            })
    return pd.DataFrame(rows)

def plot_examples(df: pd.DataFrame, events: list[dict], out: Path, tf: str, threshold: float):
    chosen=events[-min(6,len(events)):]
    if not chosen: return
    fig,axes=plt.subplots(len(chosen),1,figsize=(13,3.2*len(chosen)),squeeze=False)
    for ax,e in zip(axes[:,0],chosen):
        s=max(0,e["pre_window_start"]-5); end=min(len(df)-1,e["confirmation_index"]+max(HORIZONS))
        x=np.arange(s,end+1); o=df.open.iloc[s:end+1].to_numpy(float); h=df.high.iloc[s:end+1].to_numpy(float); l=df.low.iloc[s:end+1].to_numpy(float); c=df.close.iloc[s:end+1].to_numpy(float)
        for k in range(len(x)):
            ax.plot([x[k],x[k]],[l[k],h[k]],linewidth=0.8)
            ax.plot([x[k],x[k]],[o[k],c[k]],linewidth=4)
        ax.axvline(e["pivot_index"],linestyle="--",linewidth=1.5,label="Pivot / reversal extreme")
        ax.axvline(e["confirmation_index"],linestyle=":",linewidth=1.5,label="Confirmation candle")
        ax.set_title(f'{tf} | {threshold:.2f}% reversal | {e["event_id"]} | {e["direction"]} | after confirmation: +5={e["signed_return_5"]:.2f}% +20={e["signed_return_20"]:.2f}%')
        ax.grid(alpha=.2)
    axes[0,0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out/f"reversal_examples_{tf}_{threshold:.2f}.png",dpi=150)
    plt.close(fig)

def plot_comparison(df: pd.DataFrame, events: list[dict], matches: pd.DataFrame, out: Path, tf: str, threshold: float):
    if not events or matches.empty:
        return
    latest = events[-1]
    mm = matches[matches["event_id"] == latest["event_id"]].head(3)
    if mm.empty:
        return
    chosen = [latest]
    byid = {e["event_id"]: e for e in events}
    for mid in mm["match_event_id"]:
        if str(mid) in byid:
            chosen.append(byid[str(mid)])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 3.4 * len(chosen)), squeeze=False)
    for ax, e in zip(axes[:, 0], chosen):
        s = max(0, e["pre_window_start"] - 5)
        end = min(len(df)-1, e["confirmation_index"] + max(HORIZONS))
        o=df.open.iloc[s:end+1].to_numpy(float); h=df.high.iloc[s:end+1].to_numpy(float)
        l=df.low.iloc[s:end+1].to_numpy(float); c=df.close.iloc[s:end+1].to_numpy(float)
        base=float(c[0])
        for k in range(len(o)):
            ax.plot([k,k],[l[k]/base,h[k]/base],linewidth=.8)
            ax.plot([k,k],[o[k]/base,c[k]/base],linewidth=4)
        pivot_x=e["pivot_index"]-s; confirm_x=e["confirmation_index"]-s
        ax.axvline(pivot_x,linestyle="--",linewidth=1.4)
        ax.axvline(confirm_x,linestyle=":",linewidth=1.4)
        ax.set_ylabel("Normalized price")
        label = "CURRENT / latest" if e["event_id"] == latest["event_id"] else "HISTORICAL MATCH"
        ax.set_title(f'{label} | {e["event_id"]} | {e["direction"]} | pivot → confirmation: {e["confirmation_delay_candles"]} candles | +20 signed={e["signed_return_20"]:.2f}%')
        ax.grid(alpha=.2)
    fig.suptitle(f"{tf} | reversal structure comparison | {threshold:.2f}% threshold", y=.995)
    fig.tight_layout()
    fig.savefig(out/f"reversal_comparison_{tf}_{threshold:.2f}.png", dpi=150)
    plt.close(fig)

def plot_outcomes(summary: pd.DataFrame, out: Path, tf: str, threshold: float):
    if summary.empty:return
    fig,ax=plt.subplots(figsize=(10,5))
    x=summary.horizon.to_numpy(int)
    ax.plot(x,summary.median_signed_return.to_numpy(float),marker="o",label="Median signed return")
    ax.plot(x,summary.mean_signed_return.to_numpy(float),marker="o",label="Mean signed return")
    ax.axhline(0,linewidth=1)
    ax.set_xlabel("Candles after confirmation"); ax.set_ylabel("Movement in reversal direction (%)")
    ax.set_title(f"{tf} | {threshold:.2f}% confirmed reversals | forward behaviour")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/f"reversal_outcomes_{tf}_{threshold:.2f}.png",dpi=150); plt.close(fig)

def run(output_dir="charts",timeframes=TIMEFRAMES,thresholds=THRESHOLDS):
    out=Path(output_dir); out.mkdir(exist_ok=True)
    cfg=load_config()
    all_freq=[]; all_stats=[]
    for tf in timeframes:
        df=load_market(cfg,tf).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        feat=candle_features(df)
        for threshold in thresholds:
            turns=zigzag_turns(df,threshold)
            events=event_rows(df,turns,feat,threshold)
            ev=pd.DataFrame(events)
            stem=f"{tf}_{threshold:.2f}"
            ev.to_csv(out/f"reversal_events_{stem}.csv",index=False)
            matches=find_matches(events,feat)
            matches.to_csv(out/f"reversal_matches_{stem}.csv",index=False)
            rows=[]
            for h in HORIZONS:
                vals=pd.to_numeric(ev.get(f"signed_return_{h}",pd.Series(dtype=float)),errors="coerce").dropna()
                rows.append({"timeframe":tf,"threshold_pct":threshold,"horizon":h,"samples":len(vals),
                    "mean_signed_return":float(vals.mean()) if len(vals) else np.nan,
                    "median_signed_return":float(vals.median()) if len(vals) else np.nan,
                    "positive_count":int((vals>0).sum()) if len(vals) else 0,
                    "negative_count":int((vals<0).sum()) if len(vals) else 0,
                    "positive_share_pct":float((vals>0).mean()*100) if len(vals) else np.nan})
            sm=pd.DataFrame(rows); sm.to_csv(out/f"reversal_summary_{stem}.csv",index=False)
            plot_examples(df,events,out,tf,threshold); plot_comparison(df,events,matches,out,tf,threshold); plot_outcomes(sm,out,tf,threshold)
            sim_count=0
            if not matches.empty:
                sim_count=int((matches.similarity>=0.75).sum())
            all_freq.append({"timeframe":tf,"threshold_pct":threshold,"confirmed_reversals":len(events),"same_direction_match_rows":len(matches),"matches_similarity_ge_0.75":sim_count})
            if not ev.empty:
                for h in HORIZONS:
                    vals=pd.to_numeric(ev[f"signed_return_{h}"],errors="coerce").dropna()
                    all_stats.append({"timeframe":tf,"threshold_pct":threshold,"horizon":h,"samples":len(vals),
                        "mean_signed_return":float(vals.mean()) if len(vals) else np.nan,
                        "median_signed_return":float(vals.median()) if len(vals) else np.nan,
                        "positive_share_pct":float((vals>0).mean()*100) if len(vals) else np.nan})
            print(f"[{tf} | {threshold:.2f}%] reversals={len(events):,} | similar-match rows={len(matches):,} | >=0.75={sim_count:,}")
    pd.DataFrame(all_freq).to_csv(out/"reversal_backtest_frequency.csv",index=False)
    pd.DataFrame(all_stats).to_csv(out/"reversal_backtest_outcomes.csv",index=False)
    print("\nSaved reversal backend outputs.")
    print("Charts: reversal_examples_<tf>_<threshold>.png and reversal_outcomes_<tf>_<threshold>.png")
    print("Data: reversal_events_*, reversal_matches_*, reversal_summary_*, reversal_backtest_frequency.csv, reversal_backtest_outcomes.csv")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-timeframes",action="store_true")
    p.add_argument("--timeframes",nargs="+",choices=TIMEFRAMES,default=["1h"])
    p.add_argument("--thresholds",nargs="+",type=float,default=list(THRESHOLDS))
    p.add_argument("--output-dir",default="charts")
    a=p.parse_args()
    tfs=list(TIMEFRAMES) if a.all_timeframes else a.timeframes
    run(a.output_dir,tfs,tuple(a.thresholds))
