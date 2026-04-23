set positional-arguments

python := ".venv/bin/python"

# List available recipes.
default:
    @just --list

# Show top-level CLI help.
help:
    @{{python}} -m screener.cli --help

# Show screen command help.
help-screen:
    @{{python}} -m screener.cli screen --help

# Show historical backtest command help.
help-backtest:
    @{{python}} -m screener.cli backtest-historical --help

# Show standalone Pine strategy runner help.
help-pine:
    @{{python}} run_pinescript_strategies.py --help

# Run the screener. Example: just screen -m us -n 20 --csv
screen *args:
    @{{python}} -m screener.cli screen "$@"

# Run the US screener. Example: just screen-us -n 20 --detail
screen-us *args:
    @{{python}} -m screener.cli screen -m us "$@"

# Run the India screener. Example: just screen-india -n 20 --csv
screen-india *args:
    @{{python}} -m screener.cli screen -m india "$@"

# Run historical backtesting. Requires --as-of plus --entry/--strategy and a universe.
backtest *args:
    @{{python}} -m screener.cli backtest-historical "$@"

# Live US historical backtest smoke run.
backtest-smoke-us:
    @{{python}} -m screener.cli backtest-historical -m us --as-of 2026-03-20 --entry "close > 0" --exit false --tickers AAPL,MSFT,NVDA,AMD --hold 5 --top 2 --stop-loss 0.05 --take-profit 0.08 --trailing-stop 0.04

# Live India historical backtest smoke run.
backtest-smoke-india:
    @{{python}} -m screener.cli backtest-historical -m india --as-of 2026-03-20 --entry "close > 0" --exit false --tickers RELIANCE,TCS,INFY,HDFCBANK --hold 5 --top 2 --min-price 0 --min-avg-dollar-volume 0

# Run standalone Pine strategy backtests. Example: just pine --market us --years 3 --limit 50
pine *args:
    @{{python}} run_pinescript_strategies.py "$@"

# Run standalone Pine strategy backtests for the US market.
pine-us *args:
    @{{python}} run_pinescript_strategies.py --market us "$@"

# Run standalone Pine strategy backtests for the India market.
pine-india *args:
    @{{python}} run_pinescript_strategies.py --market india "$@"

# Compile Python files without running tests.
compile:
    @{{python}} -m compileall run_pinescript_strategies.py screener
