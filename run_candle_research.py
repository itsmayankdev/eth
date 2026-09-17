"""Run the descriptive candle-behaviour research suite.

The runner clears old chart artifacts, then produces single-timeframe state
research plus multi-timeframe research. It deliberately excludes trading,
target/stop, signal, and outcome analysis.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from candle_behavior_states import main as run_state_research
from candle_behavior_multitimeframe import main as run_multitimeframe_research


def clean_charts(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
            removed += 1
        elif item.is_dir():
            shutil.rmtree(item)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candle behaviour research suite")
    parser.add_argument("--timeframe", default="1h", help="Detailed timeframe for state research")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context", type=int, default=10)
    args = parser.parse_args()

    charts = Path("charts")
    removed = clean_charts(charts)
    print(f"Cleaned charts/: removed {removed} previous artifact(s).")

    # Reuse the existing CLIs without spawning subprocesses.
    import sys

    state_argv = [
        "candle_behavior_states.py",
        "--timeframe", args.timeframe,
        "--window", str(args.window),
        "--top-k", str(args.top_k),
        "--context", str(args.context),
    ]
    multi_argv = [
        "candle_behavior_multitimeframe.py",
        "--window", str(args.window),
        "--top-k", str(args.top_k),
    ]

    original = sys.argv
    try:
        sys.argv = state_argv
        run_state_research()
        sys.argv = multi_argv
        run_multitimeframe_research()
    finally:
        sys.argv = original

    files = sorted(p.name for p in charts.iterdir() if p.is_file())
    print("\nFINAL CHART ARTIFACTS")
    for name in files:
        print(f"  charts/{name}")


if __name__ == "__main__":
    main()
