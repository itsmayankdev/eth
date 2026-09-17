"""Backward-compatible entry point for the candle behaviour visualizer.

The project no longer uses this command for target/stop/outcome visualization.
Run ``python candle_behavior.py`` for the observational OHLC study.
"""

from candle_behavior import create_chart, main


__all__ = ["create_chart", "main"]


if __name__ == "__main__":
    main()
