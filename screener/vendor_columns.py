"""Vendor field spellings that more than one subpackage has to agree on.

A TradingView column name is not a fact about this codebase, it is a fact
about the vendor's current schema, and the vendor retires columns without
notice. When a retired column is named as a string literal in a filter, the
failure is silent in the worst way: the request succeeds, every row comes back
with ``None`` in that field, and a ``> 3`` filter therefore matches nothing.
The screen reports zero candidates, which reads exactly like "no name passed
the bar" rather than like "this screen has been broken for some time".

That is what happened to ``dividend_yield_recent``. It returned ``None`` for
every row on both markets, so ``screen -c dividend`` returned an empty frame on
both, and the ``dividend`` ranking recipe scored every name's yield as missing.

Naming each such column once, here, is what makes the next retirement a
one-line change rather than a hunt through filters, scorers, fetch lists and
display labels - which is also why this module holds no imports from the
package: the modules that need these names sit in unrelated subpackages.
"""

from __future__ import annotations

#: Trailing dividend yield in percent.
#:
#: ``dividends_yield_current`` rather than ``dividends_yield``: the latter is
#: distorted on some listings - it reports 6.37% for Alphabet's GOOGM line,
#: whose actual yield is the 0.25% ``dividends_yield_current`` gives. Replaces
#: ``dividend_yield_recent``, which the vendor retired.
DIVIDEND_YIELD_COLUMN = "dividends_yield_current"

__all__ = ["DIVIDEND_YIELD_COLUMN"]
