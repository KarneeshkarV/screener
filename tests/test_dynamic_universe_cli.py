"""Offline CLI regressions for dynamic-universe selection and rolling refill."""

from __future__ import annotations

import io
import re
import socket
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener import screen_candidates as screen_candidates_module
from screener import screen_workflow
from screener.backtester import data as backtest_data
from screener.cli import cli
from tests.conftest import StubPriceFetcher


def _flat_bars(index: pd.DatetimeIndex, *, price: float, volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
        },
        index=index,
    )


def _rising_bars(index: pd.DatetimeIndex, *, volume: float) -> pd.DataFrame:
    close = np.linspace(50.0, 100.0, len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden in this CLI regression")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def _invoke(
    config: Path,
    args: list[str],
    *,
    fetcher: StubPriceFetcher,
) -> str:
    result = CliRunner().invoke(
        cli,
        ["--config", str(config), *args],
        obj=fetcher,
        env={"SCREENER_AGENT": "0"},
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def _screen_tickers(output: str) -> list[str]:
    lines = output.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ticker,"))
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])))
    return frame["ticker"].astype(str).tolist()


def _rolling_candidate_tickers(output: str) -> list[str]:
    return re.findall(r"^\s*\d+\s+(\S+)\s+setup_score=", output, flags=re.MULTILINE)


def _period_anchor(rebalance: str, target: pd.Timestamp) -> pd.Timestamp:
    if rebalance == "daily":
        return target
    frequency = {"monthly": "M", "quarterly": "Q"}[rebalance]
    return target.to_period(frequency).start_time


@pytest.mark.parametrize(
    ("rebalance", "target_date"),
    [
        pytest.param("daily", "2024-06-03", id="daily-volume-spike"),
        pytest.param("monthly", "2024-06-28", id="monthly-anchor"),
        pytest.param("quarterly", "2024-06-28", id="quarterly-anchor"),
    ],
)
def test_screen_cli_matches_rolling_dynamic_membership_without_max_universe_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rebalance: str,
    target_date: str,
) -> None:
    """The CLI paths must rank the full base before selecting dynamic members."""
    _forbid_network(monkeypatch)
    target = pd.Timestamp(target_date)
    next_bar = pd.bdate_range(target + pd.Timedelta(days=1), periods=1)[0]
    index = pd.bdate_range("2023-01-02", next_bar)
    anchor = _period_anchor(rebalance, target)
    anchor_bar = index[index >= anchor][0]
    anchor_position = index.get_loc(anchor_bar)

    aaa = _rising_bars(index, volume=1_000.0)
    bbb = _rising_bars(index, volume=100.0)
    if rebalance == "daily":
        # Today's BBB spike must not enter the lagged ADV used for today's
        # membership. AAA remains the daily member from the two prior bars.
        bbb.loc[target, "volume"] = 10_000.0
        expected_ticker = "AAA"
        universe_symbols = "[BBB, AAA]"
    else:
        # BBB leads the completed bars used at the period anchor. AAA becomes
        # the liquidity leader later, but membership stays held through target.
        bbb.iloc[
            anchor_position - 2 : anchor_position,
            bbb.columns.get_loc("volume"),
        ] = 10_000.0
        aaa.loc[anchor_bar:, "volume"] = 20_000.0
        expected_ticker = "BBB"
        universe_symbols = "[AAA, BBB]"

    fetcher = StubPriceFetcher(
        {
            "AAA": aaa,
            "BBB": bbb,
            "SPY": _rising_bars(index, volume=1_000_000.0),
        }
    )
    monkeypatch.setattr(backtest_data, "build_price_fetcher", lambda **_kwargs: fetcher)

    class FixedDate(date):
        @classmethod
        def today(cls) -> FixedDate:
            return cls(target.year, target.month, target.day)

    monkeypatch.setattr(screen_workflow, "date", FixedDate)
    monkeypatch.setattr(screen_candidates_module, "date", FixedDate)

    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("{}\n", encoding="utf-8")
    universe_config = tmp_path / "universes.yaml"
    universe_config.write_text(
        "universes:\n"
        "  liquid_one:\n"
        "    type: dynamic\n"
        "    market: us\n"
        "    benchmark: SPY\n"
        f"    symbols: {universe_symbols}\n"
        "    size: 1\n"
        "    lookback: 2\n"
        f"    rebalance: {rebalance}\n",
        encoding="utf-8",
    )
    shared = [
        "-m",
        "us",
        "--universe",
        "liquid_one",
        "--universe-config",
        str(universe_config),
        "--max-universe",
        "1",
        "--min-price",
        "0",
        "--min-avg-dollar-volume",
        "0",
    ]

    screen_output = _invoke(
        empty_config,
        ["screen", *shared, "-c", "ema", "--sort", "setup_score", "--csv"],
        fetcher=fetcher,
    )
    rolling_output = _invoke(
        empty_config,
        [
            "backtest-rolling",
            *shared,
            "--strategy",
            "ema_stack",
            "--start",
            anchor_bar.date().isoformat(),
            "--end",
            next_bar.date().isoformat(),
            "--candidates",
        ],
        fetcher=fetcher,
    )

    assert _rolling_candidate_tickers(rolling_output) == [expected_ticker]
    assert _screen_tickers(screen_output) == [expected_ticker]


def test_rolling_cli_refill_crosses_batch_after_an_earlier_slot_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later free slot must see rank 9 after rank 8 opens from batch one."""
    _forbid_network(monkeypatch)
    index = pd.bdate_range("2023-12-01", "2024-01-05")
    data: dict[str, pd.DataFrame] = {}
    tickers: list[str] = []
    for rank in range(1, 8):
        ticker = f"EXP{rank}"
        tickers.append(ticker)
        data[ticker] = _flat_bars(
            index,
            price=2_000.0,
            volume=(11_000_000.0 - rank * 1_000_000.0) / 2_000.0,
        )
    tickers.extend(["FIRST", "SECOND"])
    data = {f"{ticker}.NS": bars for ticker, bars in data.items()}
    data["FIRST.NS"] = _flat_bars(index, price=100.0, volume=30_000.0)
    data["SECOND.NS"] = _flat_bars(index, price=100.0, volume=20_000.0)
    data["^NSEI"] = _flat_bars(index, price=100.0, volume=1_000_000.0)
    fetcher = StubPriceFetcher(data)

    config = tmp_path / "empty.yaml"
    config.write_text("{}\n", encoding="utf-8")
    output = _invoke(
        config,
        [
            "backtest-rolling",
            "-m",
            "india",
            "--tickers",
            ",".join(tickers),
            "--entry",
            "close > 0",
            "--exit",
            "false",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-05",
            "--hold",
            "10",
            "--top",
            "2",
            "--initial-capital",
            "1000",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--no-compare-reinvestment",
            "--csv",
        ],
        fetcher=fetcher,
    )

    ledger = pd.read_csv(io.StringIO(output))
    assert ledger[["ticker", "rank", "signal_date"]].to_dict("records") == [
        {"ticker": "FIRST", "rank": 8, "signal_date": "2024-01-02"},
        {"ticker": "SECOND", "rank": 9, "signal_date": "2024-01-02"},
    ]
