"""Descriptive candle-behaviour similarity and transition research.

This module is research-only. It compares candle anatomy, volatility behaviour,
and local directional behaviour, then shows context around the strongest
historical neighbours. It does not calculate entries, targets, stops, outcomes,
or predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candle_similarity import FEATURES, CandleSimilarityConfig, candle_features, candle_sequence_similarity
from config import load_config
from data.database import MarketDatabase


REGIME_FEATURES = (
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_ratio",
    "close_location",
    "direction",
    "body_change",
    "range_change",
    "alternation",
)


def load_market(cfg, timeframe):
    db = MarketDatabase(cfg.database)
    try:
        df = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()
    if df.empty:
        raise ValueError(f"No candle data available for {timeframe}")
    return df.reset_index(drop=True)


def sequence(features, end, window):
    start = end - window + 1
    if start < 0 or end >= len(features):
        raise IndexError("Invalid sequence bounds")
    return features.iloc[start : end + 1]


def rank_all(features, current_end, window, cfg):
    current_start = current_end - window + 1
    current = sequence(features, current_end, window)
    rows = []
    for end in range(window - 1, current_start):
        candidate = sequence(features, end, window)
        score = candle_sequence_similarity(current, candidate, cfg)
        rows.append({"candidate_start": end - window + 1, "candidate_end": end, "similarity": score})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    scale = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (x - med) / scale


def feature_differences(current, candidate, cfg):
    a = current[list(FEATURES)].to_numpy(float)
    b = candidate[list(FEATURES)].to_numpy(float)
    rmse = np.sqrt(np.mean((robust_z(a) - robust_z(b)) ** 2, axis=0))
    w = cfg.weights / cfg.weights.sum()
    return (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "rmse": rmse,
                "weight": w,
                "weighted_difference": rmse * w,
            }
        )
        .sort_values("weighted_difference", ascending=False)
        .reset_index(drop=True)
    )


def behaviour_state(row, range_q):
    """Assign a descriptive candle state; this is not a market forecast."""
    body = float(row.body_pct)
    upper = float(row.upper_wick_pct)
    lower = float(row.lower_wick_pct)
    rr = float(row.range_ratio)
    direction = float(row.direction)

    if rr <= range_q[0]:
        return "compression"
    if rr >= range_q[1]:
        if lower > upper * 1.35 and lower > body * 0.8:
            return "expansion_lower_rejection"
        if upper > lower * 1.35 and upper > body * 0.8:
            return "expansion_upper_rejection"
        if body >= 0.60 and direction != 0:
            return "directional_expansion"
        return "expansion"
    if lower > upper * 1.50 and lower > body:
        return "lower_rejection"
    if upper > lower * 1.50 and upper > body:
        return "upper_rejection"
    if body < 0.30:
        return "small_body"
    if direction > 0:
        return "bullish_body"
    if direction < 0:
        return "bearish_body"
    return "neutral_body"


def add_behaviour_labels(features, full_features):
    q = full_features["range_ratio"].quantile([0.25, 0.75]).to_numpy()
    out = features.copy()
    out["behaviour"] = [behaviour_state(row, q) for row in out.itertuples()]
    return out


def draw_candles(ax, df, title, match_start=None, match_end=None):
    for i, r in enumerate(df.itertuples()):
        lo = min(float(r.open), float(r.close))
        h = max(abs(float(r.close) - float(r.open)), 1e-10)
        ax.vlines(i, float(r.low), float(r.high), linewidth=0.7)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.32, lo), 0.64, h,
                fill=float(r.close) >= float(r.open), linewidth=0.5,
            )
        )

    if match_start is not None and match_end is not None:
        ax.axvspan(match_start - 0.5, match_end + 0.5, alpha=0.10)
        ax.axvline(match_start - 0.5, linestyle=":", linewidth=0.9)
        ax.axvline(match_end + 0.5, linestyle=":", linewidth=0.9)

    ax.set_xlim(-1, len(df))
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.12)


def draw_feature_panel(ax, features, title):
    x = np.arange(len(features))
    ax.plot(x, features["range_ratio"].to_numpy(), label="range")
    ax.plot(x, features["body_pct"].to_numpy(), label="body")
    ax.plot(x, features["upper_wick_pct"].to_numpy(), label="upper wick")
    ax.plot(x, features["lower_wick_pct"].to_numpy(), label="lower wick")
    ax.axhline(0.0, linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.legend(ncol=4, fontsize=7)
    ax.grid(alpha=0.12)


def transition_summary(raw, features, start, end, context):
    lo = max(0, start - context)
    hi = min(len(raw), end + context + 1)
    f = features.iloc[lo:hi].copy()
    f["index"] = np.arange(lo, hi)
    f["phase"] = "context"
    f.loc[f["index"].between(start, end), "phase"] = "matched_window"

    full_q = features["range_ratio"].quantile([0.25, 0.75]).to_numpy()
    f["behaviour"] = [behaviour_state(r, full_q) for r in f.itertuples()]

    rows = []
    for phase, g in f.groupby("phase", sort=False):
        rows.append(
            {
                "phase": phase,
                "start_index": int(g["index"].min()),
                "end_index": int(g["index"].max()),
                "mean_range_ratio": float(g["range_ratio"].mean()),
                "mean_body_pct": float(g["body_pct"].mean()),
                "bullish_fraction": float((g["direction"] > 0).mean()),
                "bearish_fraction": float((g["direction"] < 0).mean()),
                "lower_rejection_fraction": float((g["lower_wick_pct"] > g["upper_wick_pct"] * 1.5).mean()),
                "upper_rejection_fraction": float((g["upper_wick_pct"] > g["lower_wick_pct"] * 1.5).mean()),
                "behaviour_sequence": " → ".join(g["behaviour"].tolist()),
            }
        )
    return pd.DataFrame(rows), f


def create(timeframe, window, top_k, output_dir, show=False, context=10):
    cfg = load_config()
    raw = load_market(cfg, timeframe)
    features = candle_features(raw)
    sim_cfg = CandleSimilarityConfig()
    current_end = len(raw) - 1
    current_start = current_end - window + 1

    matches = rank_all(features, current_end, window, sim_cfg)
    if matches.empty:
        raise ValueError("No historical candle windows available")

    top = matches.head(top_k).copy()
    scores = matches["similarity"].to_numpy(float)
    best = float(top.iloc[0].similarity)
    current = sequence(features, current_end, window)
    best_candidate = sequence(features, int(top.iloc[0].candidate_end), window)
    contrib = feature_differences(current, best_candidate, sim_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_dir / "candle_similarity_distribution_matches.csv", index=False)
    contrib.to_csv(output_dir / "candle_similarity_feature_contributions.csv", index=False)

    # Distribution.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores, bins=60, density=True, alpha=0.75)
    ax.axvline(best, linestyle="--", linewidth=1.3, label=f"Best = {best:.4f}")
    ax.set_title("Historical candle-sequence similarity distribution")
    ax.set_xlabel("Candle similarity")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "candle_similarity_distribution.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    # Top raw candle sequences.
    n = len(top)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, max(6, 2.7 * (n + 1))), squeeze=False)
    axes = axes[:, 0]
    draw_candles(axes[0], raw.iloc[current_start : current_end + 1], f"CURRENT — last {window} candles")
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        hist = raw.iloc[int(r.candidate_start) : int(r.candidate_end) + 1]
        draw_candles(axes[rank], hist, f"#{rank} — similarity {r.similarity:.4f} — {int(r.candidate_start)}..{int(r.candidate_end)}")
    fig.suptitle(f"{cfg.symbol} {timeframe} — strongest historical candle neighbours", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "candle_similarity_top_sequences.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    # Feature contributions.
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(contrib))
    ax.bar(x, contrib.weighted_difference)
    ax.set_xticks(x)
    ax.set_xticklabels(contrib.feature, rotation=35, ha="right")
    ax.set_ylabel("Weighted normalized difference")
    ax.set_title("Candle features driving difference to best match")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "candle_similarity_feature_contributions.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    # Transition report: current window plus context before it, and the same
    # context around every top historical neighbour.
    transition_rows = []
    context_frames = []
    all_cases = [(0, current_start, current_end)] + [
        (rank, int(r.candidate_start), int(r.candidate_end))
        for rank, (_, r) in enumerate(top.iterrows(), start=1)
    ]

    for rank, start, end in all_cases:
        summary, frame = transition_summary(raw, features, start, end, context)
        frame["case"] = "CURRENT" if rank == 0 else f"HISTORICAL_{rank}"
        frame["rank"] = rank
        context_frames.append(frame)
        for row in summary.to_dict("records"):
            row["case"] = "CURRENT" if rank == 0 else f"HISTORICAL_{rank}"
            row["rank"] = rank
            if rank > 0:
                row["similarity"] = float(top.iloc[rank - 1].similarity)
                row["match_start"] = start
                row["match_end"] = end
            else:
                row["similarity"] = 1.0
                row["match_start"] = start
                row["match_end"] = end
            transition_rows.append(row)

    transition_df = pd.DataFrame(transition_rows)
    transition_df.to_csv(output_dir / "candle_transition_context_summary.csv", index=False)
    context_df = pd.concat(context_frames, ignore_index=True)
    context_df.to_csv(output_dir / "candle_transition_context_candles.csv", index=False)

    # A compact visual: current and top five, each with raw candles and the
    # behavioural feature layer. The matched window is shaded; surrounding
    # candles remain visible so the transition can be inspected directly.
    visual_top = top.head(min(5, len(top)))
    rows = 1 + len(visual_top)
    fig, axes = plt.subplots(rows * 2, 1, figsize=(15, max(10, rows * 5.0)), squeeze=False)
    axes = axes[:, 0]

    def draw_case(row_ax, feat_ax, raw_start, raw_end, label, sim):
        lo = max(0, raw_start - context)
        hi = min(len(raw), raw_end + context + 1)
        raw_ctx = raw.iloc[lo:hi]
        feat_ctx = features.iloc[lo:hi]
        ms = raw_start - lo
        me = raw_end - lo
        draw_candles(row_ax, raw_ctx, f"{label} — raw OHLC" + (f" — similarity {sim:.4f}" if sim is not None else ""), ms, me)
        draw_feature_panel(feat_ax, feat_ctx, f"{label} — body / wick / range behaviour")
        feat_ax.axvspan(ms - 0.5, me + 0.5, alpha=0.10)
        feat_ax.axvline(ms - 0.5, linestyle=":", linewidth=0.9)
        feat_ax.axvline(me + 0.5, linestyle=":", linewidth=0.9)

    draw_case(axes[0], axes[1], current_start, current_end, "CURRENT", None)
    for i, (_, r) in enumerate(visual_top.iterrows(), start=1):
        draw_case(
            axes[i * 2], axes[i * 2 + 1],
            int(r.candidate_start), int(r.candidate_end),
            f"HISTORICAL #{i}", float(r.similarity),
        )
    fig.suptitle(
        f"{cfg.symbol} {timeframe} — candle anatomy + transition context\n"
        f"Shaded region = 20-candle matched window; surrounding candles show context",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "candle_transition_context.png", dpi=160)
    if show:
        plt.show(block=False)
    plt.close(fig)

    print("\nCANDLE SIMILARITY DISTRIBUTION")
    print(f"Symbol: {cfg.symbol} | timeframe: {timeframe} | window: {window} candles")
    print(f"Historical non-overlapping windows: {len(scores)}")
    print(f"Best similarity: {best:.4f} | percentile: {np.mean(scores <= best) * 100:.2f}")
    print(
        f"Median: {np.median(scores):.4f} | P90: {np.percentile(scores, 90):.4f} | "
        f"P95: {np.percentile(scores, 95):.4f} | P99: {np.percentile(scores, 99):.4f}"
    )
    print("\nTOP HISTORICAL CANDLE SEQUENCES")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFEATURE DIFFERENCES VS BEST MATCH")
    print(contrib.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTRANSITION CONTEXT SUMMARY")
    print(transition_df[["case", "phase", "mean_range_ratio", "mean_body_pct", "bullish_fraction", "bearish_fraction", "lower_rejection_fraction", "upper_rejection_fraction"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved charts/csv outputs in", output_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("charts"))
    p.add_argument("--context", type=int, default=10, help="Candles shown before and after each matched window")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    create(**vars(a))


if __name__ == "__main__":
    main()
