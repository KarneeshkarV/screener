"""Unit tests for the shared leaf formatters in screener.format."""

from __future__ import annotations

import math

import pytest

from screener.format import (
    fmt_float,
    fmt_mcap,
    fmt_mcap_for_market,
    fmt_mcap_india,
    fmt_pct,
    fmt_volume,
)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values_render_dash(missing):
    assert fmt_float(missing) == "-"
    assert fmt_pct(missing) == "-"
    assert fmt_volume(missing) == "-"
    assert fmt_mcap(missing) == "-"
    assert fmt_mcap_india(missing) == "-"
    assert fmt_mcap_for_market(missing, "india") == "-"


def test_fmt_float_precision():
    assert fmt_float(1.2345) == "1.23"
    assert fmt_float(1.2345, 3) == "1.234"
    assert fmt_float(1.2355, 1) == "1.2"
    assert fmt_float(0) == "0.00"


def test_fmt_pct_signed_two_decimals():
    assert fmt_pct(1.5) == "+1.50%"
    assert fmt_pct(-2.0) == "-2.00%"
    assert fmt_pct(0) == "+0.00%"


def test_fmt_volume_tiers():
    assert fmt_volume(2.5e9) == "2.50B"
    assert fmt_volume(3.4e6) == "3.40M"
    assert fmt_volume(7.8e3) == "7.8K"
    assert fmt_volume(950) == "950"
    # exact tier boundaries
    assert fmt_volume(1e9) == "1.00B"
    assert fmt_volume(1e6) == "1.00M"
    assert fmt_volume(1e3) == "1.0K"


def test_fmt_mcap_tiers():
    assert fmt_mcap(2.5e12) == "2.50T"
    assert fmt_mcap(3.4e9) == "3.40B"
    assert fmt_mcap(7.8e6) == "7.8M"
    assert fmt_mcap(950_000) == "950,000"
    # exact tier boundaries
    assert fmt_mcap(1e12) == "1.00T"
    assert fmt_mcap(1e9) == "1.00B"
    assert fmt_mcap(1e6) == "1.0M"


def test_fmt_mcap_india_uses_lakh_and_crore():
    assert fmt_mcap_india(1.200652e13) == "12.01 Lakh Cr"
    assert fmt_mcap_india(6.224054e9) == "622.41 Cr"
    assert fmt_mcap_india(7.8e6) == "78.0 Lakh"
    assert fmt_mcap_india(50_000) == "50,000"
    assert fmt_mcap_india(1e12) == "1.00 Lakh Cr"
    assert fmt_mcap_india(1e7) == "1.00 Cr"
    assert fmt_mcap_india(1e5) == "1.0 Lakh"


def test_fmt_mcap_for_market_picks_india_or_us():
    assert fmt_mcap_for_market(6.224054e9, "india") == "622.41 Cr"
    assert fmt_mcap_for_market(6.224054e9, "us") == "6.22B"
    assert fmt_mcap_for_market(6.224054e9, "") == "6.22B"


def test_infinity_is_not_treated_as_missing():
    # only None / NaN are missing; inf flows through the tier logic
    assert fmt_volume(math.inf) == "infB"
