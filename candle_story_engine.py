from __future__ import annotations

"""Sequence-aware Candle Story Engine.

Research-only market-behaviour analysis. It does not create entries, targets,
stops, trade signals, or forecasts.

Architecture:
1. Fast vectorized screening over the complete history.
2. Detailed structural analysis on a small candidate pool.
3. Chronological event-sequence similarity for final historical ranking.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from price_level_interaction_v2 import analyse_window, load_market, TIMEFRAMES

DEFAULT_WINDOW = 20
DEFAULT_TOP_K = 25
SCREEN_MULTIPLIER = 4
MIN_POOL = 100

EVENT_VOCAB = [
    "INITIAL_BULLISH_BIAS", "INITIAL_BEARISH_BIAS", "INITIAL_BALANCED_BEHAVIOUR",
    "RANGE_EXPANSION", "RANGE_COMPRESSION",
    "DIRECTIONAL_SHIFT", "DIRECTION_CHANGE_IN_CHARACTER",
    "BODY_EXPANSION", "BODY_COMPRESSION", "COUNTERMOVE_AND_RENEWAL",
    "LEVEL_TEST", "LEVEL_BREAK", "HOLD", "REJECTION", "PULLBACK_TO_LEVEL",
    "REPEATED_LEVEL_TEST", "POST_TEST_EXPANSION",
]
EVENT_TO_ID = {x: i + 1 for i, x in enumerate(EVENT_VOCAB)}

STORY_FEATURES = [
    "interaction_count", "break_count", "hold_count", "rejection_count",
    "pullback_count", "repeated_level_tests", "reaction_expansion_count",
    "swing_high_tests", "swing_low_tests", "body_zone_tests",
    "first_direction", "second_direction", "direction_shift",
    "first_range", "second_range", "range_shift", "first_body", "second_body",
    "body_shift", "first_close_location", "second_close_location",
    "close_location_shift", "first_interaction_position", "last_interaction_position",
    "level_count",
]


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cs[window:] - cs[:-window]) / float(window)


def build_screen_features(df: pd.DataFrame, window: int) -> tuple[np.ndarray, list[str]]:
    o = df.open.to_numpy(float); h = df.high.to_numpy(float)
    l = df.low.to_numpy(float); c = df.close.to_numpy(float)
    n = len(df)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    direction = np.sign(c - o)
    close_loc = (c - l) / rng
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    alt = np.r_[0.0, (direction[1:] != direction[:-1]).astype(float)]
    small = (body / rng <= 0.25).astype(float)
    large = (body / rng >= 0.60).astype(float)
    upper_rej = ((upper >= 0.35) & (lower < 0.25)).astype(float)
    lower_rej = ((lower >= 0.35) & (upper < 0.25)).astype(float)
    m = n - window + 1
    if m <= 0:
        return np.empty((0, 22)), []
    half = window // 2
    def half_mean(x, second=False):
        cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
        starts = np.arange(m)
        a = starts + (half if second else 0)
        b = starts + (window if second else half)
        return (cs[b] - cs[a]) / float(half)
    vals = [half_mean(direction), half_mean(direction, True)]
    first_dir, second_dir = vals
    first_range, second_range = half_mean(rng), half_mean(rng, True)
    first_body, second_body = half_mean(body), half_mean(body, True)
    first_cl, second_cl = half_mean(close_loc), half_mean(close_loc, True)
    first_alt, second_alt = half_mean(alt), half_mean(alt, True)
    first_small, second_small = half_mean(small), half_mean(small, True)
    first_large, second_large = half_mean(large), half_mean(large, True)
    first_upper, second_upper = half_mean(upper_rej), half_mean(upper_rej, True)
    first_lower, second_lower = half_mean(lower_rej), half_mean(lower_rej, True)
    matrix = np.column_stack([
        first_dir, second_dir, second_dir-first_dir,
        first_range, second_range, second_range-first_range,
        first_body, second_body, second_body-first_body,
        first_cl, second_cl, second_cl-first_cl,
        first_alt, second_alt, first_small, second_small,
        first_large, second_large, first_upper, second_upper,
        first_lower, second_lower,
    ])
    names = [
        "first_direction","second_direction","direction_shift","first_range","second_range","range_shift",
        "first_body","second_body","body_shift","first_close_location","second_close_location","close_location_shift",
        "first_alternation","second_alternation","first_small_body","second_small_body",
        "first_large_body","second_large_body","first_upper_rejection","second_upper_rejection",
        "first_lower_rejection","second_lower_rejection"
    ]
    return matrix, names


def robust_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    x = np.vstack([query, candidates])
    med = np.median(x, axis=0); q25 = np.percentile(x,25,axis=0); q75 = np.percentile(x,75,axis=0)
    scale = np.maximum(q75-q25,1e-9)
    z = (x-med)/scale
    d = np.linalg.norm(z[1:]-z[0],axis=1)/np.sqrt(x.shape[1])
    return 1.0/(1.0+d)


def major_candle_phases(df: pd.DataFrame, start: int, end: int) -> list[str]:
    w=df.iloc[start:end+1]; o=w.open.to_numpy(float); h=w.high.to_numpy(float); l=w.low.to_numpy(float); c=w.close.to_numpy(float)
    rng=np.maximum(h-l,1e-12); body=np.abs(c-o); direction=np.sign(c-o); split=len(w)//2
    phases=[]; d1=float(np.mean(direction[:split])); d2=float(np.mean(direction[split:])); r1=float(np.mean(rng[:split])); r2=float(np.mean(rng[split:])); b1=float(np.mean(body[:split])); b2=float(np.mean(body[split:]))
    phases.append("INITIAL_BULLISH_BIAS" if d1>=0.25 else "INITIAL_BEARISH_BIAS" if d1<=-0.25 else "INITIAL_BALANCED_BEHAVIOUR")
    if r2>r1*1.20: phases.append("RANGE_EXPANSION")
    elif r2<r1*0.80: phases.append("RANGE_COMPRESSION")
    if d1*d2< -0.03 and abs(d2-d1)>=0.35: phases.append("DIRECTIONAL_SHIFT")
    elif abs(d2-d1)>=0.35: phases.append("DIRECTION_CHANGE_IN_CHARACTER")
    if b2>b1*1.20: phases.append("BODY_EXPANSION")
    elif b2<b1*0.80: phases.append("BODY_COMPRESSION")
    if len(direction)>=6:
        third=len(direction)//3; a=float(np.mean(direction[:third])); m=float(np.mean(direction[third:2*third])); z=float(np.mean(direction[2*third:]));
        if a*m< -0.05 and z*a>0.02: phases.append("COUNTERMOVE_AND_RENEWAL")
    return phases


def raw_story_tokens(df: pd.DataFrame, start: int, end: int, level_events: list[dict]) -> list[str]:
    tokens=major_candle_phases(df,start,end)
    for ev in sorted(level_events,key=lambda x:int(x["local_index"])):
        mapped={"TEST":"LEVEL_TEST","PULLBACK":"PULLBACK_TO_LEVEL","REJECTION":"REJECTION","HOLD":"HOLD","BREAK":"LEVEL_BREAK"}.get(str(ev["interaction"]),str(ev["interaction"]))
        tokens.append(mapped)
        if int(ev.get("repeated_test",0)): tokens.append("REPEATED_LEVEL_TEST")
        if int(ev.get("post_interaction_expansion",0)): tokens.append("POST_TEST_EXPANSION")
    compact=[]
    for token in tokens:
        if not compact or compact[-1]!=token: compact.append(token)
    return compact


def sequence_signature(tokens: list[str], max_len: int = 16) -> tuple[np.ndarray,np.ndarray]:
    ids=np.asarray([EVENT_TO_ID.get(t,0) for t in tokens[:max_len]],dtype=float)
    pos=np.linspace(0.0,1.0,len(ids),dtype=float) if len(ids) else np.array([],dtype=float)
    out_ids=np.zeros(max_len,dtype=float); out_pos=np.zeros(max_len,dtype=float)
    out_ids[:len(ids)]=ids; out_pos[:len(pos)]=pos
    return out_ids,out_pos


def sequence_similarity(q_tokens: list[str], c_tokens: list[str]) -> float:
    if not q_tokens or not c_tokens: return 0.0
    related={frozenset(("LEVEL_TEST","REPEATED_LEVEL_TEST")),frozenset(("RANGE_EXPANSION","BODY_EXPANSION")),frozenset(("RANGE_COMPRESSION","BODY_COMPRESSION")),frozenset(("DIRECTIONAL_SHIFT","DIRECTION_CHANGE_IN_CHARACTER"))}
    def sub_cost(a,b):
        if a==b:return 0.0
        if frozenset((a,b)) in related:return 0.35
        return 1.0
    n,m=len(q_tokens),len(c_tokens)
    dp=np.zeros((n+1,m+1),dtype=float)
    dp[:,0]=np.arange(n+1,dtype=float)
    dp[0,:]=np.arange(m+1,dtype=float)
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j]=min(dp[i-1,j]+1.0,dp[i,j-1]+1.0,dp[i-1,j-1]+sub_cost(q_tokens[i-1],c_tokens[j-1]))
    edit_sim=1.0-dp[n,m]/max(n,m)
    q_ids,q_pos=sequence_signature(q_tokens); c_ids,c_pos=sequence_signature(c_tokens)
    length=min(len(q_tokens),len(c_tokens),16)
    identity=float(np.mean(q_ids[:length]==c_ids[:length])) if length else 0.0
    spacing=1.0-float(np.mean(np.abs(q_pos[:length]-c_pos[:length]))) if length else 0.0
    return float(np.clip(0.60*edit_sim+0.25*identity+0.15*spacing,0.0,1.0))


def build_story(df: pd.DataFrame,start:int,end:int):
    level_row,level_events=analyse_window(df,start,end); tokens=raw_story_tokens(df,start,end,level_events)
    return {**level_row,"story":" -> ".join(tokens) if tokens else "NO_MEANINGFUL_STORY","story_event_count":len(level_events),"story_phase_count":len(tokens)},level_events,tokens


def detailed_feature_similarity(current:dict,candidate:dict)->float:
    q=np.asarray([float(current.get(k,0.0)) for k in STORY_FEATURES]); x=np.asarray([float(candidate.get(k,0.0)) for k in STORY_FEATURES]); scale=np.maximum(np.abs(q)*0.25,1.0); d=np.linalg.norm((x-q)/scale)/np.sqrt(len(q)); return float(1.0/(1.0+d))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--timeframe",default="1h",choices=TIMEFRAMES); p.add_argument("--all-timeframes",action="store_true"); p.add_argument("--window",type=int,default=DEFAULT_WINDOW); p.add_argument("--top-k",type=int,default=DEFAULT_TOP_K); args=p.parse_args()
    cfg=load_config(); tfs=TIMEFRAMES if args.all_timeframes else [args.timeframe]; charts=Path("charts"); charts.mkdir(exist_ok=True)
    outputs=["candle_story_current.csv","candle_story_matches.csv","candle_story_events.csv","candle_story_map.png"]
    for name in outputs:(charts/name).unlink(missing_ok=True)
    current_rows=[]; match_rows=[]; event_rows=[]
    for tf in tfs:
        df=load_market(cfg,tf)
        if len(df)<args.window*2: print(f"[{tf}] skipped: insufficient history",flush=True); continue
        print(f"[{tf}] vectorized screening ({len(df):,} candles)...",flush=True)
        screen,_=build_screen_features(df,args.window); current_idx=len(df)-args.window; current_screen=screen[-1]; candidate_screen=screen[:-1]
        coarse_sim=robust_similarity(current_screen,candidate_screen); pool_size=min(len(coarse_sim),max(args.top_k*SCREEN_MULTIPLIER,MIN_POOL)); coarse_order=np.argsort(-coarse_sim)[:pool_size]
        current,current_events,current_tokens=build_story(df,current_idx,len(df)-1); current_rows.append({"timeframe":tf,"window":args.window,**current})
        candidates=[]; candidate_indices=[]; candidate_tokens=[]
        for screen_idx in coarse_order:
            end=int(screen_idx)+args.window-1; start=end-args.window+1; row,_,tokens=build_story(df,start,end); row["screen_similarity"]=float(coarse_sim[screen_idx]); candidates.append(row); candidate_indices.append((start,end)); candidate_tokens.append(tokens)
        scored=[]
        for i,row in enumerate(candidates):
            seq=sequence_similarity(current_tokens,candidate_tokens[i]); feat=detailed_feature_similarity(current,row); score=0.75*seq+0.25*feat; scored.append((score,seq,feat,i))
        scored.sort(reverse=True)
        print(f"[{tf}] screened {len(candidate_screen):,} -> detailed {len(candidates)} -> ranked {min(args.top_k,len(scored))}",flush=True)
        for rank,(score,seq,feat,ci) in enumerate(scored[:args.top_k],1):
            row=candidates[ci]; s,e=candidate_indices[ci]; match_rows.append({"timeframe":tf,"rank":rank,"story_similarity":score,"sequence_similarity":seq,"feature_similarity":feat,"screen_similarity":row.get("screen_similarity",0.0),"candidate_start":s,"candidate_end":e,"open_time":int(df.iloc[s].open_time),"close_time":int(df.iloc[e].close_time),**row})
            _,ev,_=build_story(df,s,e)
            for event in ev:event_rows.append({"timeframe":tf,"rank":rank,"story_similarity":score,"sequence_similarity":seq,"candidate_start":s,"candidate_end":e,**event})
    pd.DataFrame(current_rows).to_csv(charts/"candle_story_current.csv",index=False); pd.DataFrame(match_rows).to_csv(charts/"candle_story_matches.csv",index=False); pd.DataFrame(event_rows).to_csv(charts/"candle_story_events.csv",index=False)
    if match_rows:
        m=pd.DataFrame(match_rows); fig,ax=plt.subplots(figsize=(14,7))
        for tf,g in m.groupby("timeframe"):
            g=g.sort_values("rank"); ax.plot(g["rank"],g["story_similarity"],marker="o",label=tf)
        ax.set_title("Sequence-Aware Candle Story Similarity"); ax.set_xlabel("Historical match rank"); ax.set_ylabel("Story similarity"); ax.grid(alpha=0.15); ax.legend(); fig.tight_layout(); fig.savefig(charts/"candle_story_map.png",dpi=160); plt.close(fig)
    print("\nCURRENT CANDLE STORIES")
    if current_rows: print(pd.DataFrame(current_rows)[["timeframe","story_event_count","story_phase_count","story"]].to_string(index=False))
    print("\nSaved:"); [print(f"charts/{x}") for x in outputs]

if __name__=="__main__": main()
