"""Error-message tests for the Pine parser."""
from __future__ import annotations

import pytest

from screener.backtester.pine import (
    PineNameError,
    PineSyntaxError,
    parse,
)


def test_empty_expression():
    with pytest.raises(PineSyntaxError, match="Empty"):
        parse("")


def test_bang_suggests_ne():
    with pytest.raises(PineSyntaxError, match="!="):
        parse("a ! b")


def test_unmatched_paren():
    with pytest.raises(PineSyntaxError):
        parse("(close > 0")


def test_trailing_garbage():
    with pytest.raises(PineSyntaxError, match="Unexpected"):
        parse("close > 0 foo")


def test_unknown_function_raises_name_error_at_eval():
    # parsing any identifier-call is accepted; name check is at evaluate() time
    # via _eval_call. Here we only assert the parser accepts and evaluate
    # raises.
    import pandas as pd

    from screener.backtester.pine import evaluate
    bars = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    ast = parse("foo(close, 5)")
    with pytest.raises(PineNameError, match="foo"):
        evaluate(ast, bars)


def test_sma_requires_integer_length():
    with pytest.raises(PineSyntaxError, match="length"):
        import pandas as pd

        from screener.backtester.pine import evaluate
        bars = pd.DataFrame(
            {
                "open": [1.0, 1.0],
                "high": [1.0, 1.0],
                "low": [1.0, 1.0],
                "close": [1.0, 1.0],
                "volume": [1.0, 1.0],
            }
        )
        evaluate(parse("sma(close, 2.5)"), bars)
