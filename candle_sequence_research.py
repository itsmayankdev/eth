from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from data.database import MarketDatabase
from structure.representation import build_structure
from structure.similarity import find_similar_structures

FEATURES = ["body_pct", "upper_wick_pct", "lower_wick_pct", "range_x", "close_loc", "efficiency"]
LABELS = ["Body %", "Upper wick %", "Lower wick %", "Range x median", "Close location %", "Efficiency"]


def prepare(df: pd.DataFrame, w: int = 12) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    c = x["close"].astype(float)

    r = (h - l).clip(lower=1e-12)
    body = (c - o).abs()
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l

    x["body_pct"] = body / r * 100
    x["upper_wick_pct"] = up / r * 100
    x["lower_wick_pct"] = lo / r * 100
    x["close_loc"] = (c - l) / r * 100
    x["signed_body"] = (c - o) / r
    x["direction"] = np.where(c >= o, 1, -1)
    x["range_pct"] = r / c * 100

    med = x["range_pct"].rolling(20, min_periods=5).median()
    x["range_x"] = x["range_pct"] / med.replace(0, np.nan)

    path = c.diff().abs().rolling(w, min_periods=max(3, w // 2)).sum()
    net = c.diff(w).abs()
    x["efficiency"] = (net / path.replace(0, np.nan)).clip(0, 1)

    rh = h.rolling(w, min_periods=max(3, w // 2)).max()
    rl = l.rolling(w, min_periods=max(3, w // 2)).min()
    rr = (rh - rl) / c * 100
    base = rr.rolling(w * 2, min_periods=max(6, w)).median()
    x["compression_ratio"] = rr / base.replace(0, np.nan)

    state = np.full(len(x), "balanced", object)
    state[((x["compression_ratio"] < 0.75) & (x["efficiency"] < 0.35)).fillna(False).to_numpy()] = "compression"
    state[(x["range_x"] > 1.5).fillna(False).to_numpy()] = "expansion"
    state[((x["range_x"] > 1.5) & (x["efficiency"] > 0.55)).fillna(False).to_numpy()] = "directional"
    state[(
        ((x["upper_wick_pct"] > 50) | (x["lower_wick_pct"] > 50))
        & (x["body_pct"] < 40)
        & (x["range_x"] > 1.15)
    ).fillna(False).to_numpy()] = "rejection"

    x["state"] = state
    x["transition"] = (
        x["range_x"].diff().abs().fillna(0)
        + x["efficiency"].diff().abs().fillna(0)
        + x["compression_ratio"].diff().abs().fillna(0)
    )
    return x


def seq(df: pd.DataFrame, center: int, radius: int) -> pd.DataFrame:
    a = max(0, center - radius)
    b = min(len(df) - 1, center + radius)
    z = df.iloc[a : b + 1].copy()
    z["offset"] = np.arange(a - center, b - center + 1)
    return z


def candles(ax, z: pd.DataFrame) -> None:
    for i, r in enumerate(z.itertuples()):
        ax.vlines(i, r.low, r.high, lw=0.8)
        lo = min(r.open, r.close)
        ht = max(abs(r.close - r.open), 1e-9)
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, ht, fill=r.close >= r.open, lw=0.6))
    ax.set_xlim(-1, len(z))
    ax.grid(alpha=0.12)


def _candle_index_from_structure_row(row: pd.Series) -> int:
    """Return the candle position, never the pandas row Index object."""
    return int(row["index"])


def create(
    timeframe: str,
    threshold: float,
    pivots: int,
    top_k: int,
    minimum_similarity: float,
    radius: int,
    output_dir: Path,
    show: bool = False,
):
    cfg = load_config()
    db = MarketDatabase(cfg.database)
    try:
        df = db.load_candles(cfg.symbol, timeframe)
    finally:
        db.close()

    if df.empty:
        raise ValueError(f"No candle data available for {timeframe}")

    df = prepare(df)
    structure = build_structure(df, threshold)
    if len(structure) < pivots * 2:
        raise ValueError("Not enough pivots")

    matches = find_similar_structures(
        structure,
        n_pivots=pivots,
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )
    if matches.empty:
        raise ValueError("No historical matches")

    cur_end = len(structure) - 1
    items = [("CURRENT", cur_end, None)] + [
        (f"MATCH #{i + 1}  sim {float(m['similarity']):.4f}", int(m["candidate_end_position"]), m)
        for i, (_, m) in enumerate(matches.iterrows())
    ]

    fig, axs = plt.subplots(
        len(items),
        2,
        figsize=(18, 3.8 * len(items)),
        squeeze=False,
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    records = []

    for row, (name, endpos, match) in enumerate(items):
        spos = endpos - pivots + 1
        p0 = _candle_index_from_structure_row(structure.iloc[spos])
        p1 = _candle_index_from_structure_row(structure.iloc[endpos])
        z = df.iloc[max(0, p0 - radius * 2) : min(len(df), p1 + radius + 1)]

        ax = axs[row, 0]
        candles(ax, z)
        z_start = int(z.index[0]) if len(z) else 0
        for _, p in structure.iloc[spos : endpos + 1].iterrows():
            candle_idx = _candle_index_from_structure_row(p)
            xx = candle_idx - z_start
            if 0 <= xx < len(z):
                ax.axvline(xx, ls=":", lw=0.6, alpha=0.4)
                ax.annotate(
                    str(p["pivot_type"]),
                    (xx, float(p["price"])),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

        sequence = " → ".join(structure.iloc[spos : endpos + 1]["pivot_type"].astype(str))
        ax.set_title(name + " — actual candles | " + sequence)
        ax.set_ylabel("USDT")

        q = seq(df, p1, radius)
        vals = q[FEATURES].to_numpy(float).T
        finite = vals[np.isfinite(vals)]
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        im = axs[row, 1].imshow(
            vals,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=max(vmax, 1),
        )
        axs[row, 1].set_yticks(range(len(LABELS)))
        axs[row, 1].set_yticklabels(LABELS, fontsize=8)
        axs[row, 1].set_xticks(range(len(q)))
        axs[row, 1].set_xticklabels(q["offset"].astype(int), fontsize=7)
        axs[row, 1].set_xlabel("Candles relative to final pivot")
        axs[row, 1].set_title("Raw candle anatomy")
        fig.colorbar(im, ax=axs[row, 1], fraction=0.046, pad=0.03)

        for _, r in q.iterrows():
            records.append(
                {
                    "sequence": name,
                    "similarity": None if match is None else float(match["similarity"]),
                    "offset": int(r["offset"]),
                    **{f: float(r[f]) for f in FEATURES},
                    "state": str(r["state"]),
                }
            )

    fig.suptitle(
        f"{cfg.symbol} {timeframe} — Candle Sequence Research\n"
        "Raw OHLC + candle anatomy around structural endpoints. Descriptive only; no trades or predictions.",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    a = output_dir / "candle_sequence_research.png"
    fig.savefig(a, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig2, axs2 = plt.subplots(
        len(matches),
        2,
        figsize=(16, 3.2 * len(matches)),
        squeeze=False,
    )
    cur = seq(df, _candle_index_from_structure_row(structure.iloc[cur_end]), radius)

    for i, (_, match) in enumerate(matches.iterrows()):
        hist_end = _candle_index_from_structure_row(
            structure.iloc[int(match["candidate_end_position"])]
        )
        hist = seq(df, hist_end, radius)
        for j, z in enumerate((cur, hist)):
            base_idx = min(radius, len(z) - 1)
            base = float(z["close"].iloc[base_idx])
            path = (z["close"] / base - 1) * 100
            axs2[i, j].plot(z["offset"], path, lw=1.4)
            axs2[i, j].axvline(0, ls="--", lw=0.7, alpha=0.45)
            axs2[i, j].axhline(0, lw=0.5, alpha=0.25)
            axs2[i, j].grid(alpha=0.12)
            axs2[i, j].set_xlabel("Relative candle")
            axs2[i, j].set_ylabel("Close change from center %")
            axs2[i, j].set_title(
                "CURRENT" if j == 0 else f"MATCH #{i + 1} — sim {float(match['similarity']):.4f}"
            )

    fig2.suptitle(
        f"{cfg.symbol} {timeframe} — Current vs Historical Candle Sequence Microscope",
        fontsize=14,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    b = output_dir / "candle_sequence_microscope.png"
    fig2.savefig(b, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    csv_path = output_dir / "candle_sequence_microscope.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return a, b


def main() -> None:
    p = argparse.ArgumentParser(description="Research ETH candle sequences around structural events.")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--pivots", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--minimum-similarity", type=float, default=0.70)
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("charts"))
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    x, y = create(**vars(a))
    print(f"Saved: {x}\nSaved: {y}\nSaved: {a.output_dir / 'candle_sequence_microscope.csv'}")


if __name__ == "__main__":
    main()
