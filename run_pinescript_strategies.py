"""Compatibility wrapper for the standalone Pine strategy runner."""

from screener.backtester.pine_runner import *  # noqa: F401,F403
from screener.backtester.pine_runner import main


if __name__ == "__main__":
    main()
