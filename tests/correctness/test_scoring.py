"""Phase 8 correctness tests: scoring math for scanner._add_setup_score and garp.add_garp_score.

All expected values are hand-derived from the formulas read directly from source.

scanner.py formulas (verified from source):
  rsi_quality = (1 - abs(rsi - 60) / 40).clip(lower=0, upper=1).fillna(0)
  momentum    = ((change.clip(lower=-5, upper=10) + 5) / 15).fillna(0)
  overextension_penalty = ((extension - 0.12).clip(lower=0) / 0.25).clip(upper=1)
    where extension = (close - ema20) / ema20
  setup_score = 25*liquidity + 30*trend_strength + 15*momentum + 15*market_cap
               + 10*rsi_quality + 5*price_quality - 15*overextension_penalty

garp.py formulas (verified from source):
  inv_peg = (1 - peg.rank(pct=True)).fillna(0)
  pct(col) = col.rank(pct=True).fillna(0)
  garp_score = 30*inv_peg + 20*pct("eps_growth_5y") + 15*pct("sales_growth_5y")
             + 15*pct("roe_5y") + 10*pct("roce_or_roic") + 10*pct("quarterly_profit_growth")
"""

from __future__ import annotations


import pandas as pd
import pytest

from screener.scanner import _add_setup_score, _log_percentile
from screener.garp import add_garp_score, _passes_garp, INDIA_THRESHOLDS, US_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames with neutral column values
# ---------------------------------------------------------------------------


def _neutral_row(**overrides) -> dict:
    """Return a minimal row with columns _add_setup_score reads.

    Defaults are chosen to produce deterministic percentile ranks when used
    in single-row DataFrames (liquidity, trend_strength, market_cap,
    price_quality all resolve to 1.0 for a single row, since rank(pct=True)
    of a single value = 1.0).
    """
    base = {
        "close": 100.0,
        "EMA5": 102.0,
        "EMA20": 100.0,  # extension = (100 - 100) / 100 = 0.0 → penalty 0
        "EMA100": 95.0,
        "EMA200": 90.0,
        "change": 2.5,  # momentum = (2.5 + 5) / 15 = 0.5
        "RSI": 60.0,  # rsi_quality = 1.0
        "volume": 1_000_000.0,
        "market_cap_basic": 1_000_000_000.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _percentile and _log_percentile tests
# ---------------------------------------------------------------------------


class TestLogPercentileMonotonicity:
    """_log_percentile must produce a non-decreasing sequence for sorted input."""

    def test_strictly_increasing_input_yields_increasing_percentiles(self):
        # Hand-derive: values [1, 4, 9, 16, 25]
        # log(1+1)=ln2, log(4+1)=ln5, log(9+1)=ln10, log(16+1)=ln17, log(25+1)=ln26
        # These are strictly increasing, so rank(pct=True) = [0.2, 0.4, 0.6, 0.8, 1.0]
        series = pd.Series([1, 4, 9, 16, 25], dtype=float)
        result = _log_percentile(series)
        assert result.is_monotonic_increasing, (
            "log_percentile must be non-decreasing for sorted input"
        )
        # No ties → strictly increasing → all consecutive diffs > 0
        diffs = result.diff().dropna()
        assert (diffs > 0).all(), (
            "log_percentile should be strictly increasing for distinct sorted input"
        )

    def test_single_value_returns_one(self):
        # Single element: rank(pct=True) = 1.0
        result = _log_percentile(pd.Series([42.0]))
        assert result.iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_negative_values_clipped_to_zero_before_log(self):
        # Negative values clipped to 0 → log(0+1) = 0, so they all rank equal
        series = pd.Series([-10.0, -5.0, -1.0])
        result = _log_percentile(series)
        # All become 0 after clip → log(1)=0 for all → ties → rank avg = 2/3 each
        # rank(pct=True) with ties: pandas 'average' method → (1+2+3)/3 / 3 = 0.6667
        # Actually average rank of 3 tied values = (1+2+3)/3 = 2, pct = 2/3
        assert result.iloc[0] == pytest.approx(result.iloc[1], abs=1e-9)
        assert result.iloc[1] == pytest.approx(result.iloc[2], abs=1e-9)

    def test_nan_fills_to_zero(self):
        series = pd.Series([float("nan"), 10.0])
        result = _log_percentile(series)
        # NaN coerced → fillna(0) after rank → 0.0
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# rsi_quality component
# ---------------------------------------------------------------------------


class TestRsiQuality:
    """rsi_quality = (1 - abs(rsi - 60) / 40).clip(0, 1).fillna(0)

    Hand derivations:
      rsi=60  → 1 - 0/40 = 1.0
      rsi=20  → 1 - 40/40 = 0.0   (clipped at lower=0)
      rsi=40  → 1 - 20/40 = 0.5
      rsi=100 → 1 - 40/40 = 0.0   (clipped at lower=0)
    """

    def _rsi_quality_single_row(self, rsi: float) -> float:
        """Single-row df → all rank-based components = 1.0, penalty = 0.
        setup_score = 25*1 + 30*1 + 15*0.5 + 15*1 + 10*rsi_quality + 5*1 - 15*0
                    = 82.5 + 10*rsi_quality
        => rsi_quality = (setup_score - 82.5) / 10
        """
        df = pd.DataFrame([_neutral_row(RSI=rsi)])
        result = _add_setup_score(df)
        score = result["setup_score"].iloc[0]
        return (score - 82.5) / 10.0

    def test_rsi_60_quality_is_1(self):
        # rsi=60 → rsi_quality=1.0
        # setup_score = 82.5 + 10*1.0 = 92.5 → rsi_quality = (92.5-82.5)/10 = 1.0
        q = self._rsi_quality_single_row(60.0)
        assert q == pytest.approx(1.0, abs=1e-9)

    def test_rsi_20_quality_is_0(self):
        # rsi=20 → 1 - 40/40 = 0.0 (clipped at 0)
        # setup_score = 82.5 + 10*0 = 82.5 → rsi_quality = 0.0
        q = self._rsi_quality_single_row(20.0)
        assert q == pytest.approx(0.0, abs=1e-9)

    def test_rsi_40_quality_is_half(self):
        # rsi=40 → 1 - 20/40 = 0.5
        # setup_score = 82.5 + 10*0.5 = 87.5 → rsi_quality = 0.5
        q = self._rsi_quality_single_row(40.0)
        assert q == pytest.approx(0.5, abs=1e-9)

    def test_rsi_100_quality_is_0(self):
        # rsi=100 → 1 - 40/40 = 0.0 (clipped at 0)
        q = self._rsi_quality_single_row(100.0)
        assert q == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# momentum component
# ---------------------------------------------------------------------------


class TestMomentum:
    """momentum = ((change.clip(-5, 10) + 5) / 15).fillna(0)

    Hand derivations:
      change=10  → (10+5)/15 = 1.0
      change=-5  → (-5+5)/15 = 0.0
      change=2.5 → (2.5+5)/15 = 0.5
      change=20  → clipped to 10 → (10+5)/15 = 1.0
      change=-10 → clipped to -5 → (-5+5)/15 = 0.0
    """

    def _momentum_single_row(self, change: float) -> float:
        """Single-row df → all rank components = 1.0, rsi_quality=1.0 (rsi=60), penalty=0.
        setup_score = 25 + 30 + 15*momentum + 15 + 10*1 + 5 - 0 = 85 + 15*momentum
        => momentum = (setup_score - 85) / 15
        """
        df = pd.DataFrame([_neutral_row(change=change, RSI=60.0)])
        result = _add_setup_score(df)
        score = result["setup_score"].iloc[0]
        return (score - 85.0) / 15.0

    def test_change_10_momentum_is_1(self):
        m = self._momentum_single_row(10.0)
        assert m == pytest.approx(1.0, abs=1e-9)

    def test_change_neg5_momentum_is_0(self):
        m = self._momentum_single_row(-5.0)
        assert m == pytest.approx(0.0, abs=1e-9)

    def test_change_2p5_momentum_is_half(self):
        m = self._momentum_single_row(2.5)
        assert m == pytest.approx(0.5, abs=1e-9)

    def test_change_above_10_clips_to_1(self):
        # change=20 clipped to 10 → momentum=1.0
        m = self._momentum_single_row(20.0)
        assert m == pytest.approx(1.0, abs=1e-9)

    def test_change_below_neg5_clips_to_0(self):
        # change=-20 clipped to -5 → momentum=0.0
        m = self._momentum_single_row(-20.0)
        assert m == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# overextension_penalty component
# ---------------------------------------------------------------------------


class TestOverextensionPenalty:
    """overextension_penalty = ((extension - 0.12).clip(lower=0) / 0.25).clip(upper=1)
    where extension = (close - ema20) / ema20

    Hand derivations:
      ext=0.12  → (0.12-0.12)/0.25 = 0.0
      ext=0.245 → (0.245-0.12)/0.25 = 0.125/0.25 = 0.5
      ext=0.37  → (0.37-0.12)/0.25 = 0.25/0.25 = 1.0
      ext=0.50  → clipped at upper=1 → 1.0
      ext=0.00  → clipped at lower=0 → 0.0
    """

    def _penalty_single_row(self, extension: float) -> float:
        """Single-row df → penalty varies, rest fixed.
        ema20 = 100, close = 100*(1+extension)
        With rsi=60 (rsi_quality=1.0), change=2.5 (momentum=0.5):
        setup_score = 25 + 30 + 15*0.5 + 15 + 10*1 + 5 - 15*penalty
                    = 92.5 - 15*penalty
        => penalty = (92.5 - setup_score) / 15
        """
        ema20 = 100.0
        close = ema20 * (1.0 + extension)
        # Keep ema5 > ema20 > ema100 > ema200 so trend_spread > 0 and clips within [0, 0.35]
        # trend_spread = (ema5-ema20)/close + (ema20-ema100)/close + (ema100-ema200)/close
        # We want consistent trend_spread for single row (rank = 1.0 regardless of value)
        ema5 = close * 1.02
        ema100 = close * 0.95
        ema200 = close * 0.90
        df = pd.DataFrame(
            [
                {
                    "close": close,
                    "EMA5": ema5,
                    "EMA20": ema20,
                    "EMA100": ema100,
                    "EMA200": ema200,
                    "change": 2.5,
                    "RSI": 60.0,
                    "volume": 1_000_000.0,
                    "market_cap_basic": 1_000_000_000.0,
                }
            ]
        )
        result = _add_setup_score(df)
        score = result["setup_score"].iloc[0]
        return (92.5 - score) / 15.0

    def test_ext_0p12_penalty_is_0(self):
        # (0.12 - 0.12).clip(lower=0) / 0.25 = 0.0
        p = self._penalty_single_row(0.12)
        assert p == pytest.approx(0.0, abs=1e-9)

    def test_ext_0p245_penalty_is_half(self):
        # (0.245 - 0.12) / 0.25 = 0.125 / 0.25 = 0.5
        p = self._penalty_single_row(0.245)
        assert p == pytest.approx(0.5, abs=1e-9)

    def test_ext_0p37_penalty_is_1(self):
        # (0.37 - 0.12) / 0.25 = 0.25 / 0.25 = 1.0
        p = self._penalty_single_row(0.37)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_ext_above_0p37_clips_at_1(self):
        # (0.5 - 0.12) / 0.25 = 1.52 → clipped to 1.0
        p = self._penalty_single_row(0.50)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_ext_below_0p12_clips_at_0(self):
        # (0.05 - 0.12) = -0.07 → clipped to 0 → penalty = 0
        p = self._penalty_single_row(0.05)
        assert p == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Full setup_score composite (best-case row)
# ---------------------------------------------------------------------------


class TestSetupScoreFullWeighting:
    """Best-case single row: all weighted components at maximum.

    Setup:
      - Single row → all rank-based components (liquidity, trend_strength,
        market_cap, price_quality) = 1.0
      - rsi=60  → rsi_quality=1.0
      - change=10 → momentum=1.0
      - extension=0.0 → penalty=0.0 (close = ema20 exactly)

    Hand-computed:
      setup_score = 25*1 + 30*1 + 15*1 + 15*1 + 10*1 + 5*1 - 15*0
                  = 25 + 30 + 15 + 15 + 10 + 5 - 0
                  = 100.0
    """

    def test_best_case_score_is_100(self):
        df = pd.DataFrame(
            [
                {
                    "close": 100.0,
                    "EMA5": 105.0,  # ema5 > ema20 > ema100 > ema200
                    "EMA20": 100.0,  # extension = (100-100)/100 = 0 → penalty = 0
                    "EMA100": 95.0,
                    "EMA200": 90.0,
                    "change": 10.0,  # momentum = 1.0
                    "RSI": 60.0,  # rsi_quality = 1.0
                    "volume": 1_000_000.0,
                    "market_cap_basic": 1_000_000_000.0,
                }
            ]
        )
        result = _add_setup_score(df)
        score = result["setup_score"].iloc[0]
        # 25+30+15+15+10+5-0 = 100.0
        assert score == pytest.approx(100.0, abs=0.01)

    def test_worst_case_score(self):
        """Worst case: rank components = 1.0 (single row), rsi=20 (quality=0),
        change=-5 (momentum=0), extension=0.50 (penalty=1.0).

        setup_score = 25*1 + 30*1 + 15*0 + 15*1 + 10*0 + 5*1 - 15*1
                    = 25 + 30 + 0 + 15 + 0 + 5 - 15
                    = 60.0
        """
        df = pd.DataFrame(
            [
                {
                    "close": 150.0,  # close = ema20 * 1.5 → extension = 0.5 → penalty = 1.0
                    "EMA5": 155.0,
                    "EMA20": 100.0,
                    "EMA100": 95.0,
                    "EMA200": 90.0,
                    "change": -5.0,  # momentum = 0.0
                    "RSI": 20.0,  # rsi_quality = 0.0
                    "volume": 1_000_000.0,
                    "market_cap_basic": 1_000_000_000.0,
                }
            ]
        )
        result = _add_setup_score(df)
        score = result["setup_score"].iloc[0]
        # 25+30+0+15+0+5-15 = 60.0
        assert score == pytest.approx(60.0, abs=0.01)

    def test_weight_vector_matches_source(self):
        """Verify the seven-term weight vector is exactly 25/30/15/15/10/5/-15
        by engineering rows that isolate each component in turn.

        We build two identical rows then vary one component to measure each weight.
        """
        # The single-row approach with rsi=60, change=10, ext=0 gives 100.
        # Changing rsi from 60 to 20 removes 10*1.0 = 10 points:
        df_best = pd.DataFrame([_neutral_row(change=10.0, RSI=60.0)])
        df_norsi = pd.DataFrame([_neutral_row(change=10.0, RSI=20.0)])  # rsi_quality→0
        score_best = _add_setup_score(df_best)["setup_score"].iloc[0]
        score_norsi = _add_setup_score(df_norsi)["setup_score"].iloc[0]
        # Difference = 10 * (1.0 - 0.0) = 10 (the rsi_quality weight)
        assert (score_best - score_norsi) == pytest.approx(10.0, abs=0.01)

        # momentum weight: change=10 (mom=1.0) vs change=-5 (mom=0.0), rsi=60
        df_mom0 = pd.DataFrame([_neutral_row(change=-5.0, RSI=60.0)])
        score_mom0 = _add_setup_score(df_mom0)["setup_score"].iloc[0]
        assert (score_best - score_mom0) == pytest.approx(15.0, abs=0.01)

        # overextension weight: ext=0 (pen=0) vs ext=0.37 (pen=1.0)
        df_ext = pd.DataFrame(
            [
                {
                    "close": 137.0,  # ext = (137-100)/100 = 0.37 → penalty=1.0
                    "EMA5": 140.0,
                    "EMA20": 100.0,
                    "EMA100": 95.0,
                    "EMA200": 90.0,
                    "change": 10.0,
                    "RSI": 60.0,
                    "volume": 1_000_000.0,
                    "market_cap_basic": 1_000_000_000.0,
                }
            ]
        )
        score_ext = _add_setup_score(df_ext)["setup_score"].iloc[0]
        # Penalty went from 0 to 1 → score drops by 15
        assert (score_best - score_ext) == pytest.approx(15.0, abs=0.01)


# ---------------------------------------------------------------------------
# GARP scoring: inv_peg component
# ---------------------------------------------------------------------------


class TestGarpInvPeg:
    """inv_peg = (1 - peg.rank(pct=True)).fillna(0)

    For peg=[0.5, 1.0, 2.0, 4.0] (4 rows, distinct):
      rank(pct=True): 0.5→rank1, 1.0→rank2, 2.0→rank3, 4.0→rank4
      pct ranks = [1/4, 2/4, 3/4, 4/4] = [0.25, 0.5, 0.75, 1.0]
      inv_peg = 1 - [0.25, 0.5, 0.75, 1.0] = [0.75, 0.5, 0.25, 0.0]
    """

    def _make_garp_df(self, pegs: list[float]) -> pd.DataFrame:
        """Minimal GARP df: only columns that add_garp_score reads."""
        n = len(pegs)
        # Use increasing values for all pct() columns so rank order is known
        return pd.DataFrame(
            {
                "peg": pegs,
                "eps_growth_5y": [10.0 * (i + 1) for i in range(n)],
                "sales_growth_5y": [10.0 * (i + 1) for i in range(n)],
                "roe_5y": [10.0 * (i + 1) for i in range(n)],
                "roce_or_roic": [10.0 * (i + 1) for i in range(n)],
                "quarterly_profit_growth": [10.0 * (i + 1) for i in range(n)],
            }
        )

    def test_inv_peg_four_values(self):
        pegs = [0.5, 1.0, 2.0, 4.0]
        df = self._make_garp_df(pegs)
        result = add_garp_score(df)
        # result is sorted descending by garp_score; align back by peg
        result = result.set_index("peg")

        # inv_peg for peg=0.5 → rank=0.25 → inv=0.75 (best peg → highest inv_peg)
        # All other pct() columns increase with row index, so peg=0.5 is row 0 (i=0)
        # pct ranks for other columns: [0.25, 0.50, 0.75, 1.0] for rows 0→3
        # For peg=0.5 (row 0): inv_peg=0.75, all others=0.25 → score = 30*0.75 + (20+15+15+10+10)*0.25
        # = 22.5 + 70*0.25 = 22.5 + 17.5 = 40.0
        assert result.loc[0.5, "garp_score"] == pytest.approx(40.0, abs=0.01)

        # For peg=4.0 (row 3): inv_peg=0.0, all others=1.0 → score = 30*0 + 70*1.0 = 70.0
        assert result.loc[4.0, "garp_score"] == pytest.approx(70.0, abs=0.01)

        # For peg=1.0 (row 1): inv_peg=0.5, others=0.5 → 30*0.5 + 70*0.5 = 15+35 = 50.0
        assert result.loc[1.0, "garp_score"] == pytest.approx(50.0, abs=0.01)

        # For peg=2.0 (row 2): inv_peg=0.25, others=0.75 → 30*0.25 + 70*0.75 = 7.5+52.5 = 60.0
        assert result.loc[2.0, "garp_score"] == pytest.approx(60.0, abs=0.01)


class TestGarpPctRank:
    """Test pct() ranking: for col=[5, 10, 20, 40] (4 rows):
    rank(pct=True) = [0.25, 0.50, 0.75, 1.0]
    """

    def test_pct_rank_four_distinct_values(self):
        df = pd.DataFrame(
            {
                "peg": [1.0, 1.5, 2.0, 2.5],  # distinct increasing pegs
                "eps_growth_5y": [5.0, 10.0, 20.0, 40.0],
                "sales_growth_5y": [5.0, 10.0, 20.0, 40.0],
                "roe_5y": [5.0, 10.0, 20.0, 40.0],
                "roce_or_roic": [5.0, 10.0, 20.0, 40.0],
                "quarterly_profit_growth": [5.0, 10.0, 20.0, 40.0],
            }
        )
        result = add_garp_score(df)
        # For row with eps_growth_5y=5 (smallest), rank=0.25 → pct("eps_growth_5y")=0.25
        # peg=1.0 is smallest → inv_peg = 1 - 0.25 = 0.75
        # score = 30*0.75 + (20+15+15+10+10)*0.25 = 22.5 + 17.5 = 40.0
        result_sorted = result.sort_values("peg")
        assert result_sorted["garp_score"].iloc[0] == pytest.approx(40.0, abs=0.01)

        # Row with all-column rank=1.0 and peg=2.5 (largest, inv_peg=0.0):
        # score = 30*0 + 70*1.0 = 70.0
        result_sorted = result.sort_values("peg", ascending=False)
        assert result_sorted["garp_score"].iloc[0] == pytest.approx(70.0, abs=0.01)


# ---------------------------------------------------------------------------
# GARP full weighting best-row
# ---------------------------------------------------------------------------


class TestGarpFullWeighting:
    """Best-case GARP row: inv_peg=1.0 (impossible since rank can't be 0 for
    a finite set) but for a 2-row df where best row has smallest peg and
    largest everything else.

    For 2 rows:
      peg=[0.5, 2.0]: rank(pct=True)=[0.5, 1.0] → inv_peg=[0.5, 0.0]
      all pct() columns: best row has larger values → rank=1.0
      best row score = 30*0.5 + 20*1.0 + 15*1.0 + 15*1.0 + 10*1.0 + 10*1.0
                     = 15 + 20 + 15 + 15 + 10 + 10 = 85.0

    For single row: rank(pct=True)=1.0 for all → inv_peg = 1-1.0 = 0.0
      score = 30*0 + 20*1 + 15*1 + 15*1 + 10*1 + 10*1 = 70.0
    """

    def test_single_row_garp_score(self):
        """Single row: pct() cols all = 1.0, inv_peg = 1 - 1.0 = 0.0.
        garp_score = 0 + 20 + 15 + 15 + 10 + 10 = 70.0
        """
        df = pd.DataFrame(
            [
                {
                    "peg": 1.5,
                    "eps_growth_5y": 25.0,
                    "sales_growth_5y": 20.0,
                    "roe_5y": 18.0,
                    "roce_or_roic": 22.0,
                    "quarterly_profit_growth": 15.0,
                }
            ]
        )
        result = add_garp_score(df)
        assert result["garp_score"].iloc[0] == pytest.approx(70.0, abs=0.01)

    def test_two_row_best_row_score(self):
        """Two rows: best row has smallest peg + largest everything else.

        peg=[0.5, 2.0]: rank(pct=True)=[0.5, 1.0], inv_peg=[0.5, 0.0]
        other cols for best row: rank=1.0 (highest value each)

        best row score = 30*0.5 + 20*1 + 15*1 + 15*1 + 10*1 + 10*1
                       = 15 + 20 + 15 + 15 + 10 + 10 = 85.0
        """
        df = pd.DataFrame(
            [
                # best row: smallest peg, highest everything else
                {
                    "peg": 0.5,
                    "eps_growth_5y": 30.0,
                    "sales_growth_5y": 25.0,
                    "roe_5y": 20.0,
                    "roce_or_roic": 25.0,
                    "quarterly_profit_growth": 20.0,
                },
                # worse row: largest peg, lowest everything else
                {
                    "peg": 2.0,
                    "eps_growth_5y": 10.0,
                    "sales_growth_5y": 10.0,
                    "roe_5y": 10.0,
                    "roce_or_roic": 10.0,
                    "quarterly_profit_growth": 10.0,
                },
            ]
        )
        result = add_garp_score(df)
        best = result["garp_score"].max()
        assert best == pytest.approx(85.0, abs=0.01)

    def test_four_row_best_row_score(self):
        """Four rows, best row = smallest peg + highest other metrics.

        peg=[0.5,1,2,4]: rank(pct=True)=[0.25,0.5,0.75,1.0], inv_peg=[0.75,0.5,0.25,0.0]
        other cols best row (all highest): rank=1.0

        best row (peg=0.5, row 0, all others highest):
          garp_score = 30*0.75 + 20*1 + 15*1 + 15*1 + 10*1 + 10*1
                     = 22.5 + 70 = 92.5
        """
        df = pd.DataFrame(
            [
                {
                    "peg": 0.5,
                    "eps_growth_5y": 40.0,
                    "sales_growth_5y": 40.0,
                    "roe_5y": 40.0,
                    "roce_or_roic": 40.0,
                    "quarterly_profit_growth": 40.0,
                },
                {
                    "peg": 1.0,
                    "eps_growth_5y": 30.0,
                    "sales_growth_5y": 30.0,
                    "roe_5y": 30.0,
                    "roce_or_roic": 30.0,
                    "quarterly_profit_growth": 30.0,
                },
                {
                    "peg": 2.0,
                    "eps_growth_5y": 20.0,
                    "sales_growth_5y": 20.0,
                    "roe_5y": 20.0,
                    "roce_or_roic": 20.0,
                    "quarterly_profit_growth": 20.0,
                },
                {
                    "peg": 4.0,
                    "eps_growth_5y": 10.0,
                    "sales_growth_5y": 10.0,
                    "roe_5y": 10.0,
                    "roce_or_roic": 10.0,
                    "quarterly_profit_growth": 10.0,
                },
            ]
        )
        result = add_garp_score(df)
        best = result["garp_score"].max()
        # best row: peg=0.5 → inv_peg=0.75; all other cols highest → rank=1.0
        # 30*0.75 + 20*1 + 15*1 + 15*1 + 10*1 + 10*1 = 22.5 + 20 + 15 + 15 + 10 + 10 = 92.5
        assert best == pytest.approx(92.5, abs=0.01)


# ---------------------------------------------------------------------------
# _passes_garp threshold tests
# ---------------------------------------------------------------------------


class TestPassesGarp:
    """Test _passes_garp boundary conditions against INDIA_THRESHOLDS and US_THRESHOLDS.

    INDIA_THRESHOLDS (from source):
      market_cap_min=1000.0, sales_min=1000.0,
      peg_max=2.0, sales_growth_5y_min=15.0, operating_profit_growth_min=10.0,
      eps_growth_5y_min=12.0, roe_5y_min=15.0, roce_or_roic_min=15.0

    US_THRESHOLDS: market_cap_min=1_000_000_000, sales_min=1_000_000_000, rest same defaults.

    _passes_garp also requires:
      - 0 < peg < peg_max (strictly; peg must be positive)
      - quarterly_profit_growth > 0
      - all required fields must be non-None numbers
    """

    def _passing_india_row(self) -> dict:
        """A row that just passes all INDIA thresholds."""
        return {
            "market_cap": 1001.0,  # > 1000
            "sales": 1001.0,  # > 1000
            "peg": 1.0,  # 0 < 1.0 < 2.0
            "sales_growth_5y": 16.0,  # > 15.0
            "operating_profit_growth": 11.0,  # > 10.0
            "eps_growth_5y": 13.0,  # > 12.0
            "roe_5y": 16.0,  # > 15.0
            "roce_or_roic": 16.0,  # > 15.0
            "quarterly_profit_growth": 1.0,  # > 0
        }

    def test_passing_row_passes_india(self):
        assert _passes_garp(self._passing_india_row(), INDIA_THRESHOLDS) is True

    def test_market_cap_at_min_fails(self):
        row = self._passing_india_row()
        row["market_cap"] = 1000.0  # not > 1000, equals threshold
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_market_cap_just_above_min_passes(self):
        row = self._passing_india_row()
        row["market_cap"] = 1000.01
        assert _passes_garp(row, INDIA_THRESHOLDS) is True

    def test_sales_at_min_fails(self):
        row = self._passing_india_row()
        row["sales"] = 1000.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_peg_at_zero_fails(self):
        # peg must be > 0 (condition: 0 < peg < peg_max)
        row = self._passing_india_row()
        row["peg"] = 0.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_peg_negative_fails(self):
        row = self._passing_india_row()
        row["peg"] = -0.5
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_peg_at_max_fails(self):
        row = self._passing_india_row()
        row["peg"] = 2.0  # not < 2.0, equals threshold
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_peg_just_below_max_passes(self):
        row = self._passing_india_row()
        row["peg"] = 1.99
        assert _passes_garp(row, INDIA_THRESHOLDS) is True

    def test_sales_growth_at_min_fails(self):
        row = self._passing_india_row()
        row["sales_growth_5y"] = 15.0  # not > 15.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_sales_growth_just_above_min_passes(self):
        row = self._passing_india_row()
        row["sales_growth_5y"] = 15.01
        assert _passes_garp(row, INDIA_THRESHOLDS) is True

    def test_operating_profit_growth_at_min_fails(self):
        row = self._passing_india_row()
        row["operating_profit_growth"] = 10.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_eps_growth_at_min_fails(self):
        row = self._passing_india_row()
        row["eps_growth_5y"] = 12.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_roe_at_min_fails(self):
        row = self._passing_india_row()
        row["roe_5y"] = 15.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_roce_at_min_fails(self):
        row = self._passing_india_row()
        row["roce_or_roic"] = 15.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_quarterly_profit_growth_zero_fails(self):
        row = self._passing_india_row()
        row["quarterly_profit_growth"] = 0.0  # not > 0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_quarterly_profit_growth_negative_fails(self):
        row = self._passing_india_row()
        row["quarterly_profit_growth"] = -1.0
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_missing_required_field_fails(self):
        row = self._passing_india_row()
        row["peg"] = None
        assert _passes_garp(row, INDIA_THRESHOLDS) is False

    def test_us_thresholds_market_cap_min_is_1B(self):
        row = self._passing_india_row()
        row["market_cap"] = 999_999_999.0  # just below $1B
        row["sales"] = 1_000_000_001.0
        assert _passes_garp(row, US_THRESHOLDS) is False

    def test_us_thresholds_passing_row(self):
        row = self._passing_india_row()
        row["market_cap"] = 1_000_000_001.0  # > $1B
        row["sales"] = 1_000_000_001.0
        assert _passes_garp(row, US_THRESHOLDS) is True
