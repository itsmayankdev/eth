"""Historical multi-timeframe candle-behaviour relationship research.

Descriptive research only. Historical windows are compared across the same
reference time. No trades, targets, stops, outcomes, or predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from candle_similarity import FEATURES, candle_features
from data.database import MarketDatabase

TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
DEFAULT_STEP = 20
ANCHOR_TF = "1h"
TF_MS = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000}


def feature_vector(features: pd.DataFrame, start: int, window: int) -> np.ndarray:
    v = features[list(FEATURES)].to_numpy(float)[start:start + window]
    i = {n: k for k, n in enumerate(FEATURES)}
    d, body, rng = v[:,i["direction"]], v[:,i["body_pct"]], v[:,i["range_ratio"]]
    upper, lower = v[:,i["upper_wick_pct"]], v[:,i["lower_wick_pct"]]
    close, alt = v[:,i["close_location"]], v[:,i["alternation"]]
    h = window // 2
    return np.array([d.mean(),body.mean(),rng.mean(),upper.mean(),lower.mean(),close.mean(),alt.mean(),d[:h].mean(),d[h:].mean(),rng[:h].mean(),rng[h:].mean(),body[:h].mean(),body[h:].mean(),close[:h].mean(),close[h:].mean(),alt[:h].mean(),alt[h:].mean(),rng[h:].mean()-rng[:h].mean(),d[h:].mean()-d[:h].mean()])


def build_history(features: pd.DataFrame, window: int):
    """Vectorized construction of all rolling window summaries."""
    a = features[list(FEATURES)].to_numpy(float)
    n = len(a)
    starts = np.arange(n-window+1, dtype=int)
    h = window // 2
    cs = np.vstack([np.zeros((1,a.shape[1])), np.cumsum(a,axis=0)])
    total = (cs[starts+window]-cs[starts])/window
    first = (cs[starts+h]-cs[starts])/h
    second = (cs[starts+window]-cs[starts+h])/h
    idx = {n:k for k,n in enumerate(FEATURES)}
    d,b,r,u,l,c,al = [idx[x] for x in ("direction","body_pct","range_ratio","upper_wick_pct","lower_wick_pct","close_location","alternation")]
    return np.column_stack((total[:,d],total[:,b],total[:,r],total[:,u],total[:,l],total[:,c],total[:,al],first[:,d],second[:,d],first[:,r],second[:,r],first[:,b],second[:,b],first[:,c],second[:,c],first[:,al],second[:,al],second[:,r]-first[:,r],second[:,d]-first[:,d])), starts


def robust_scale(matrix):
    med=np.nanmedian(matrix,axis=0); scale=np.nanmedian(np.abs(matrix-med),axis=0)*1.4826
    return med,np.where(scale<1e-6,1.0,scale)


def similarity(current, history):
    med,scale=robust_scale(history)
    return np.exp(-np.sqrt(np.mean(((history-med)/scale-(current-med)[None,:]/scale)**2,axis=1)))


def align_index(close_times: np.ndarray, target: int, timeframe: str):
    """Return latest completed window ending at or before reference time."""
    pos=int(np.searchsorted(close_times,target,side="right"))-1
    if pos<0: return None
    if int(target)-int(close_times[pos]) > TF_MS[timeframe]*1.25: return None
    return pos


def run(window, top_k, step, output_dir):
    if window<4 or window%2: raise ValueError("window must be an even number >= 4")
    if step<1: raise ValueError("step must be >= 1")
    out=Path(output_dir); out.mkdir(exist_ok=True)
    db=MarketDatabase("data/eth_market.db"); data={}

    for tf in TIMEFRAMES:
        df=db.load_candles("ETHUSDT",tf)
        if len(df)<window: continue
        df=df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        feats=candle_features(df)
        matrix,starts=build_history(feats,window)
        current=feature_vector(feats,len(feats)-window,window)
        sims=similarity(current,matrix)
        ends=starts+window-1
        opens=df.iloc[ends]["open_time"].to_numpy(dtype=np.int64)
        closes=opens+TF_MS[tf]
        order=np.argsort(closes)
        closes=closes[order]; opens=opens[order]; sims=sims[order]; starts=starts[order]
        data[tf]={"df":df,"starts":starts,"opens":opens,"closes":closes,"current":current,"similarity":sims}
        print(f"[{tf}] {len(matrix)} full windows prepared")

    if ANCHOR_TF not in data: raise RuntimeError("1h data is required")
    current_rows=[]
    for tf in TIMEFRAMES:
        if tf not in data: continue
        x=data[tf]["current"]
        current_rows.append({"timeframe":tf,"direction_balance":x[0],"body_level":x[1],"range_level":x[2],"upper_wick":x[3],"lower_wick":x[4],"close_location":x[5],"alternation":x[6],"direction_shift":x[18],"range_shift":x[17]})
    current_df=pd.DataFrame(current_rows)

    common_start=max(int(x["closes"][0]) for x in data.values())
    common_end=min(int(x["closes"][-1]) for x in data.values())
    if common_start>common_end: raise RuntimeError("No common historical overlap")

    anchor=data[ANCHOR_TF]
    anchor_ref=anchor["closes"]
    current_anchor_start=len(anchor["df"])-window
    mask=(anchor_ref>=common_start)&(anchor_ref<=common_end)&(anchor["starts"]!=current_anchor_start)&((anchor["starts"]%step)==0)
    candidates=np.flatnonzero(mask)
    print(f"Common completed-candle overlap: {common_start} -> {common_end}")
    print(f"Synchronized 1h reference candidates: {len(candidates)}")
    if not len(candidates): raise RuntimeError("No synchronized historical candidates")

    ranked=candidates[np.argsort(anchor["similarity"][candidates])[::-1]]
    results=[]; skipped=0
    for rank,i in enumerate(ranked,1):
        ref=int(anchor_ref[i]); row={"anchor_rank_1h":rank,"reference_timestamp":ref,"anchor_timestamp_1h":int(anchor["opens"][i])}; scores=[]; ok=True
        for tf in TIMEFRAMES:
            if tf not in data: ok=False; break
            x=data[tf]; j=align_index(x["closes"],ref,tf)
            if j is None: ok=False; break
            row[f"{tf}_similarity"]=float(x["similarity"][j]); row[f"{tf}_timestamp"]=int(x["opens"][j]); row[f"{tf}_close_timestamp"]=int(x["closes"][j]); scores.append(float(x["similarity"][j]))
        if not ok: skipped+=1; continue
        gaps=[ref-row[f"{tf}_close_timestamp"] for tf in TIMEFRAMES]
        row["combined_similarity"]=float(np.mean(scores)); row["max_alignment_gap_ms"]=max(gaps); row["alignment_status"]="SYNCHRONIZED"; results.append(row)
        if len(results)>=max(top_k,25): break

    if not results: raise RuntimeError("No synchronized historical multi-timeframe windows found after alignment")
    matches=pd.DataFrame(results).sort_values("combined_similarity",ascending=False).head(top_k).reset_index(drop=True)
    current_df["research_note"]="Descriptive candle-behaviour comparison only"
    matches["research_note"]="Completed-candle synchronized historical comparison; not a forecast"
    current_df.to_csv(out/"candle_mtf_history_current.csv",index=False); matches.to_csv(out/"candle_mtf_history_matches.csv",index=False)
    print("\nMULTI-TIMEFRAME SYNCHRONIZED HISTORICAL CANDLE BEHAVIOUR RESEARCH\n")
    print(current_df.to_string(index=False)); print("\nTOP SYNCHRONIZED HISTORICAL WINDOWS\n")
    print(matches[["anchor_rank_1h","reference_timestamp","combined_similarity","max_alignment_gap_ms"]+[f"{tf}_similarity" for tf in TIMEFRAMES]].to_string(index=False))
    print(f"\nCandidates skipped during alignment: {skipped}"); print("\nSaved:"); print(out/"candle_mtf_history_current.csv"); print(out/"candle_mtf_history_matches.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--window",type=int,default=DEFAULT_WINDOW); p.add_argument("--top-k",type=int,default=DEFAULT_TOP_K); p.add_argument("--step",type=int,default=DEFAULT_STEP); p.add_argument("--output-dir",default="charts"); a=p.parse_args(); run(a.window,a.top_k,a.step,a.output_dir)
