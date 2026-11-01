"""The dividend yield column is one spelling everywhere it is used.

``screen -c dividend`` returned zero rows on both markets for as long as the
vendor had been retiring ``dividend_yield_recent``. The request still
succeeded, every row came back with ``None`` in that field, and ``> 3``
therefore matched nothing - so the screen reported "no candidates", which
reads like an answer rather than like a broken screen.

The spelling reaches four places that have to agree: the TradingView filter,
the ranking recipe's declared columns, the scan's detail fetch list, and the
display label. These tests pin that agreement, because a column named as a
literal in only three of the four fails the same silent way.
"""

from __future__ import annotations

from screener.criteria import resolve_criteria
from screener.display import COLUMNS
from screener.scanner import DETAIL_COLUMNS
from screener.scoring import get_scorer
from screener.vendor_columns import DIVIDEND_YIELD_COLUMN


def test_the_dividend_filter_names_the_shared_column():
    rendered = str(resolve_criteria(("dividend",)).filters)
    assert DIVIDEND_YIELD_COLUMN in rendered


def test_the_dividend_scorer_fetches_the_shared_column():
    assert DIVIDEND_YIELD_COLUMN in get_scorer("dividend").columns


def test_the_detail_fetch_includes_the_shared_column():
    """``--detail`` must fetch the column the display is prepared to label."""
    assert DIVIDEND_YIELD_COLUMN in DETAIL_COLUMNS


def test_the_display_labels_the_shared_column():
    assert COLUMNS[DIVIDEND_YIELD_COLUMN].label == "Div%"


def test_the_retired_spelling_is_gone_from_every_seam():
    """The retired name returns ``None`` for every row on both markets.

    Left anywhere in a filter it silently empties the screen, so it must not
    survive in any of the four seams.
    """
    retired = "dividend_yield_recent"
    assert retired not in str(resolve_criteria(("dividend",)).filters)
    assert retired not in get_scorer("dividend").columns
    assert retired not in DETAIL_COLUMNS
    assert retired not in COLUMNS
