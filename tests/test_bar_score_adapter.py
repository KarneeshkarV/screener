"""The screen-side bar-score adapter must not hide an outage or an adjustment mismatch.

Two failure modes are pinned here, both of which used to be silent:

* A NaN score has two causes that look identical in the output. Too little
  history is a real eligibility verdict; an empty fetch is a price-provider
  outage. Since unscored rows are dropped, a total outage rendered as "0
  results", which reads exactly like "nothing matched your filters".
* ``score_bars`` reads ``close``, so a dividend-adjusted screen close and a
  ``--price-adjustment splits_only`` backtest close produce two different
  numbers for one factor name. That is the drift this layer exists to remove,
  and a single synthetic frame cannot catch it: the two sides have to be fed
  the frames their own adjustment mode would really return.

Everything here is offline; the price fetcher is a fake.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd
import pytest

import screener.backtester.data as backtester_data
from screener.factors import get_price_score, score_bars
from screener.scoring import OUTPUT_SCORE_COLUMN
from screener.scoring import bar_scores as bar_scores_module
from screener.scoring.bar_scores import apply_bar_score, bar_scores_for_tickers

_LOGGER_NAME = "screener.scoring.bar_scores"
_MARKET = "india"
_SPEC = get_price_score("momentum_12_1")
_INDEX = pd.bdate_range("2022-01-03", periods=320)
_AS_OF = _INDEX[-1].date()


def _bars(
    start: float, daily_growth: float, periods: int = len(_INDEX)
) -> pd.DataFrame:
    """A deterministic geometric trend, so 12-1 momentum is exactly known."""
    index = _INDEX[-periods:]
    close = pd.Series(
        start * (1.0 + daily_growth) ** np.arange(periods, dtype=float), index=index
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": pd.Series(1_000_000.0, index=index),
        }
    )


class _FakeFetcher:
    """Serves canned frames; an absent symbol comes back as an empty frame."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data
        self.requested: list[list[str]] = []

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        symbols = list(tickers)
        self.requested.append(symbols)
        return {s: self._data.get(s, pd.DataFrame()) for s in symbols}


def _rows(*tv_tickers: str) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": tv, "name": tv} for tv in tv_tickers])


# --- finding 4: a fetch outage must not look like an empty screen ------------


def test_partial_price_outage_warns_and_names_the_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = _FakeFetcher({"ALPHA.NS": _bars(100.0, 0.002)})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA", "NSE:BETA", "NSE:GAMMA"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert not np.isnan(scores["NSE:ALPHA"])
    assert np.isnan(scores["NSE:BETA"]) and np.isnan(scores["NSE:GAMMA"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2/3" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_total_price_outage_is_logged_at_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = _FakeFetcher({})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA", "NSE:BETA"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert all(np.isnan(v) for v in scores.values())
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "2/2" in message
    assert "outage" in message


def test_short_history_is_not_reported_as_a_fetch_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A name with real but insufficient bars is ineligible, not an outage."""
    fetcher = _FakeFetcher(
        {
            "ALPHA.NS": _bars(100.0, 0.002),
            "BETA.NS": _bars(50.0, 0.001, periods=60),
        }
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA", "NSE:BETA"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert np.isnan(scores["NSE:BETA"])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "1/2" in infos[0].getMessage()


def test_a_frame_without_a_close_column_counts_as_no_price_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken = _bars(100.0, 0.002).drop(columns=["close"])
    fetcher = _FakeFetcher({"ALPHA.NS": broken})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA"], _SPEC, market=_MARKET, as_of=_AS_OF, fetcher=fetcher
        )

    assert np.isnan(scores["NSE:ALPHA"])
    assert [r for r in caplog.records if r.levelno == logging.ERROR]


def test_unscored_rows_are_still_dropped_after_an_outage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The NaN-means-ineligible drop policy is unchanged; only the logging is new."""
    fetcher = _FakeFetcher({"ALPHA.NS": _bars(100.0, 0.002)})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scored = apply_bar_score(
            _rows("NSE:ALPHA", "NSE:BETA"),
            _SPEC,
            market=_MARKET,
            output_column=OUTPUT_SCORE_COLUMN,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert scored["ticker"].tolist() == ["NSE:ALPHA"]
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_a_fully_scored_scan_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    fetcher = _FakeFetcher(
        {"ALPHA.NS": _bars(100.0, 0.002), "BETA.NS": _bars(80.0, 0.001)}
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        bar_scores_for_tickers(
            ["NSE:ALPHA", "NSE:BETA"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# --- finding 3: setup_score stays 0-100; raw recipe value lives in aux ------


def test_setup_score_is_the_percentile_of_the_raw_recipe_value() -> None:
    """``setup_score`` stays on the scale ``execution-trade`` already thresholds."""
    fetcher = _FakeFetcher(
        {"ALPHA.NS": _bars(100.0, 0.002), "BETA.NS": _bars(80.0, 0.001)}
    )
    scored = apply_bar_score(
        _rows("NSE:ALPHA", "NSE:BETA"),
        _SPEC,
        market=_MARKET,
        output_column=OUTPUT_SCORE_COLUMN,
        as_of=_AS_OF,
        fetcher=fetcher,
    ).set_index("ticker")

    raw_alpha = float(score_bars(_SPEC, _bars(100.0, 0.002)).iloc[-1])
    raw_beta = float(score_bars(_SPEC, _bars(80.0, 0.001)).iloc[-1])
    assert raw_alpha > raw_beta > 0
    assert scored.loc["NSE:ALPHA", "mom_12_1"] == pytest.approx(raw_alpha)
    assert scored.loc["NSE:BETA", "mom_12_1"] == pytest.approx(raw_beta)
    assert scored.loc["NSE:ALPHA", OUTPUT_SCORE_COLUMN] == pytest.approx(100.0)
    assert scored.loc["NSE:BETA", OUTPUT_SCORE_COLUMN] == pytest.approx(50.0)


def test_non_positive_momentum_is_dropped_before_the_percentile() -> None:
    """A loser is ineligible, not the worst-ranked name still in the table."""
    fetcher = _FakeFetcher(
        {"WIN.NS": _bars(100.0, 0.002), "LOSE.NS": _bars(100.0, -0.001)}
    )
    scored = apply_bar_score(
        _rows("NSE:WIN", "NSE:LOSE"),
        _SPEC,
        market=_MARKET,
        output_column=OUTPUT_SCORE_COLUMN,
        as_of=_AS_OF,
        fetcher=fetcher,
    )

    assert scored["ticker"].tolist() == ["NSE:WIN"]
    assert float(scored["mom_12_1"].iloc[0]) > 0
    # A lone survivor is the 100th percentile of a one-name field.
    assert scored[OUTPUT_SCORE_COLUMN].iloc[0] == pytest.approx(100.0)


# --- finding 5: the screen must adjust closes the way the backtest does ------


@pytest.fixture
def adjustment_probe(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Capture the kwargs the adapter passes to ``build_price_fetcher``.

    ``bar_scores`` imports the builder inside the function, so the patch has to
    land on the defining module rather than on ``bar_scores`` itself.
    """
    calls: list[dict[str, object]] = []

    def _fake_builder(**kwargs: object) -> _FakeFetcher:
        calls.append(kwargs)
        adjusted = bool(kwargs.get("auto_adjust"))
        # Dividend-adjusted closes are strictly below raw closes here, so the
        # two adjustment modes cannot accidentally agree.
        return _FakeFetcher({"ALPHA.NS": _bars(100.0, 0.002 if adjusted else 0.001)})

    monkeypatch.setattr(backtester_data, "build_price_fetcher", _fake_builder)
    return calls


def test_default_adjustment_states_the_screens_assumption(
    adjustment_probe: list[dict[str, object]],
) -> None:
    """The default is the backtester's own default: fully adjusted closes."""
    assert bar_scores_module.DEFAULT_PRICE_ADJUSTMENT == "full"
    bar_scores_for_tickers(["NSE:ALPHA"], _SPEC, market=_MARKET, as_of=_AS_OF)
    assert adjustment_probe[0]["auto_adjust"] is True


@pytest.mark.parametrize(
    ("price_adjustment", "auto_adjust"),
    [("full", True), ("splits_only", False), ("none", False)],
)
def test_adjustment_maps_to_auto_adjust_exactly_as_the_backtester_does(
    adjustment_probe: list[dict[str, object]],
    price_adjustment: str,
    auto_adjust: bool,
) -> None:
    """Mirrors ``build_backtest_fetcher``: only ``full`` sets ``auto_adjust``."""
    bar_scores_for_tickers(
        ["NSE:ALPHA"],
        _SPEC,
        market=_MARKET,
        as_of=_AS_OF,
        price_adjustment=price_adjustment,  # type: ignore[arg-type]
    )
    assert adjustment_probe[0]["auto_adjust"] is auto_adjust


def test_apply_bar_score_threads_the_adjustment_through(
    adjustment_probe: list[dict[str, object]],
) -> None:
    apply_bar_score(
        _rows("NSE:ALPHA"),
        _SPEC,
        market=_MARKET,
        output_column=OUTPUT_SCORE_COLUMN,
        as_of=_AS_OF,
        price_adjustment="splits_only",
    )
    assert adjustment_probe[0]["auto_adjust"] is False


def test_splits_only_screen_score_matches_split_adjusted_backtest_score() -> None:
    """A split in the momentum window must not create a false screen return."""
    raw_bars = _bars(100.0, 0.001)
    split_row = 160
    raw_bars.loc[raw_bars.index[:split_row], ["open", "high", "low", "close"]] *= 2
    raw_bars["split_factor"] = 1.0
    raw_bars.loc[raw_bars.index[:split_row], "split_factor"] = 2.0
    fetcher = _FakeFetcher({"ALPHA.NS": raw_bars})

    adjusted_bars = backtester_data.apply_splits_only_adjustment(
        {"ALPHA.NS": raw_bars}
    )["ALPHA.NS"]
    backtest_score = float(score_bars(_SPEC, adjusted_bars).iloc[-1])
    raw_score = float(score_bars(_SPEC, raw_bars).iloc[-1])
    assert raw_score != pytest.approx(backtest_score)

    screen_score = bar_scores_for_tickers(
        ["NSE:ALPHA"],
        _SPEC,
        market=_MARKET,
        as_of=_AS_OF,
        fetcher=fetcher,
        price_adjustment="splits_only",
    )["NSE:ALPHA"]

    assert screen_score == pytest.approx(backtest_score)


def test_an_injected_fetcher_keeps_its_own_adjustment(
    adjustment_probe: list[dict[str, object]],
) -> None:
    """A caller-supplied fetcher is used as given; nothing is rebuilt behind it."""
    fetcher = _FakeFetcher({"ALPHA.NS": _bars(100.0, 0.003)})
    scores = bar_scores_for_tickers(
        ["NSE:ALPHA"],
        _SPEC,
        market=_MARKET,
        as_of=_AS_OF,
        fetcher=fetcher,
        price_adjustment="none",
    )

    assert adjustment_probe == []
    assert scores["NSE:ALPHA"] == pytest.approx(
        float(score_bars(_SPEC, _bars(100.0, 0.003)).iloc[-1])
    )


def test_an_unknown_adjustment_spelling_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown price_adjustment"):
        bar_scores_for_tickers(
            ["NSE:ALPHA"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            price_adjustment="adjusted",  # type: ignore[arg-type]
        )


def test_an_empty_ticker_list_short_circuits_before_any_fetch(
    adjustment_probe: list[dict[str, object]],
) -> None:
    assert bar_scores_for_tickers([], _SPEC, market=_MARKET, as_of=_AS_OF) == {}
    assert adjustment_probe == []


# --- floor drops: rows failing eligible_above are counted and named ----------


def test_below_floor_rows_are_logged_at_info_as_eligibility_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A value that fails the recipe's floor is logged as its own drop reason."""
    fetcher = _FakeFetcher(
        {"WIN.NS": _bars(100.0, 0.002), "LOSE.NS": _bars(100.0, -0.001)}
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        apply_bar_score(
            _rows("NSE:WIN", "NSE:LOSE"),
            _SPEC,
            market=_MARKET,
            output_column=OUTPUT_SCORE_COLUMN,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    message = infos[0].getMessage()
    assert "1/2" in message
    assert "momentum_12_1" in message
    assert "(0)" in message


def test_a_floor_that_removes_every_candidate_warns_not_silences(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An all-dropped scan must say why instead of printing an empty table."""
    fetcher = _FakeFetcher(
        {"LOSE.NS": _bars(100.0, -0.001), "WORSE.NS": _bars(100.0, -0.002)}
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scored = apply_bar_score(
            _rows("NSE:LOSE", "NSE:WORSE"),
            _SPEC,
            market=_MARKET,
            output_column=OUTPUT_SCORE_COLUMN,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert scored.empty
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage().lower()
    assert "no name passed the recipe's floor" in message
    assert '"nothing matched your filters"' in message
    assert "2/2" in message
    assert "momentum_12_1" in message
    assert "(0)" in message


def test_a_recipe_without_a_floor_never_logs_floor_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No ``eligible_above`` declaration means no floor verdicts to report."""
    spec = get_price_score("momentum_12_1")
    bare = spec.__class__(
        name=spec.name,
        score_fn=spec.score_fn,
        required_lookback=spec.required_lookback,
        description=spec.description,
        aux_column=spec.aux_column,
        eligible_above=None,
    )
    fetcher = _FakeFetcher({"ALPHA.NS": _bars(100.0, -0.001)})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        apply_bar_score(
            _rows("NSE:ALPHA"),
            bare,
            market=_MARKET,
            output_column=OUTPUT_SCORE_COLUMN,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# --- stale last bars: a dead listing must not rank as current ----------------


def test_a_fresh_last_bar_scores_normally(caplog: pytest.LogCaptureFixture) -> None:
    """Bars ending at ``as_of`` are current. The value is the raw momentum."""
    bars = _bars(100.0, 0.002)
    fetcher = _FakeFetcher({"ALPHA.NS": bars})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA"], _SPEC, market=_MARKET, as_of=_AS_OF, fetcher=fetcher
        )

    assert scores["NSE:ALPHA"] == pytest.approx(float(score_bars(_SPEC, bars).iloc[-1]))
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


def test_a_last_bar_60_days_before_as_of_yields_nan_and_logs_stale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A long frame whose coverage stopped is dropped as untradeable, not ranked.

    The fresh name next to it must still score normally. The stale count is
    kept apart from short history and from live values.
    """
    stale_bars = _bars(80.0, 0.001)
    stale_bars.index = stale_bars.index - pd.Timedelta(days=60)
    fetcher = _FakeFetcher({"ALPHA.NS": _bars(100.0, 0.002), "OLD.NS": stale_bars})
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        scores = bar_scores_for_tickers(
            ["NSE:ALPHA", "NSE:OLD"],
            _SPEC,
            market=_MARKET,
            as_of=_AS_OF,
            fetcher=fetcher,
        )

    assert not np.isnan(scores["NSE:ALPHA"])
    assert np.isnan(scores["NSE:OLD"])

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    message = infos[0].getMessage()
    assert "1/2" in message
    assert "stale last bar" in message
    assert "untradeable" in message
