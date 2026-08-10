"""Safe Pine-like expression parser and evaluator.

Parses a small subset of Pine Script into an AST and evaluates it against a
pandas DataFrame of OHLCV bars. Does NOT use Python ``eval``.

Two evaluation entry points:

  :func:`evaluate`      one ticker's bars -> one Series.
  :func:`evaluate_panel` many tickers at once -> ``{ticker: Series}``. Tickers
                        that share a bar index and column set are stacked into
                        column-per-ticker frames and the AST is walked **once**
                        over the whole panel, so each node costs one pandas call
                        instead of one per ticker. Because only exactly-aligned
                        tickers are grouped, no reindex padding is introduced and
                        the per-column values are identical to evaluating each
                        ticker on its own.

Supported:
  series: open, high, low, close, volume, adj_close, plus numeric columns
          joined onto the bars (for example options-panel fields such as pcr)
  literals: int / float
  arithmetic: + - * /
  comparison: > >= < <= == !=
  boolean:    and, or, not
  functions:  sma(s, n), ema(s, n), rsi(s, n), highest(s, n), lowest(s, n),
              atr(n), crossover(a, b), crossunder(a, b)

Causality guarantee: every rolling / shift operation is left-aligned, so the
value at bar ``i`` depends only on bars ``<= i``. crossover/crossunder use a
single ``.shift(1)`` and so compare only bars ``i`` and ``i-1``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from screener.indicators.frames import wilder_atr, wilder_rsi


class PineError(Exception):
    """Base class for Pine expression errors."""


class PineSyntaxError(PineError):
    """Raised when the expression fails to tokenize or parse."""


class PineNameError(PineError):
    """Raised when an unknown identifier or function is referenced."""


SERIES_NAMES = {"open", "high", "low", "close", "volume", "adj_close"}
ROLLING_FUNCS = {"sma", "ema", "rsi", "highest", "lowest"}
FUNC_NAMES = ROLLING_FUNCS | {"atr", "crossover", "crossunder"}
BOOL_KEYWORDS = {"and", "or", "not", "true", "false"}


# ── AST ──────────────────────────────────────────────────────────────


class Num(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float


class Name(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class UnaryOp(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["-", "+"]
    operand: Node


class Not(BaseModel):
    model_config = ConfigDict(frozen=True)

    operand: Node


class BinOp(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["+", "-", "*", "/"]
    left: Node
    right: Node


class Compare(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal[">", ">=", "<", "<=", "==", "!="]
    left: Node
    right: Node


class BoolOp(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["and", "or"]
    left: Node
    right: Node


class Call(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    args: list[Node]
    col: int  # column in source (for error messages)


# In-memory construction only — no discriminator, so model_validate from dicts is intentionally not supported.
Node = Union[Num, Name, UnaryOp, Not, BinOp, Compare, BoolOp, Call]

UnaryOp.model_rebuild()
Not.model_rebuild()
BinOp.model_rebuild()
Compare.model_rebuild()
BoolOp.model_rebuild()
Call.model_rebuild()


# ── tokenizer ────────────────────────────────────────────────────────


_TWO_CHAR = {">=", "<=", "==", "!="}
_SINGLE_PUNCT = set("+-*/()><,")


class Token(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["num", "name", "op", "lp", "rp", "comma", "end"]
    value: str
    col: int


def _tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            start = i
            saw_dot = ch == "."
            i += 1
            while i < n and (expr[i].isdigit() or (expr[i] == "." and not saw_dot)):
                if expr[i] == ".":
                    saw_dot = True
                i += 1
            tokens.append(Token(kind="num", value=expr[start:i], col=start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            tokens.append(Token(kind="name", value=expr[start:i], col=start))
            continue
        two = expr[i : i + 2]
        if two in _TWO_CHAR:
            tokens.append(Token(kind="op", value=two, col=i))
            i += 2
            continue
        if ch == "(":
            tokens.append(Token(kind="lp", value=ch, col=i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(kind="rp", value=ch, col=i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(kind="comma", value=ch, col=i))
            i += 1
            continue
        if ch in _SINGLE_PUNCT:
            tokens.append(Token(kind="op", value=ch, col=i))
            i += 1
            continue
        raise PineSyntaxError(f"Unexpected character {ch!r} at column {i}")
    tokens.append(Token(kind="end", value="", col=n))
    return tokens


# ── parser (recursive descent with precedence climbing) ─────────────


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = value or kind
            raise PineSyntaxError(
                f"Expected {expected!r} at column {tok.col}, got {tok.value!r}"
            )
        return self.consume()

    # entry point
    def parse(self) -> Node:
        node = self.parse_or()
        if self.peek().kind != "end":
            tok = self.peek()
            raise PineSyntaxError(f"Unexpected {tok.value!r} at column {tok.col}")
        return node

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.peek().kind == "name" and self.peek().value == "or":
            self.consume()
            right = self.parse_and()
            left = BoolOp(op="or", left=left, right=right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while self.peek().kind == "name" and self.peek().value == "and":
            self.consume()
            right = self.parse_not()
            left = BoolOp(op="and", left=left, right=right)
        return left

    def parse_not(self) -> Node:
        if self.peek().kind == "name" and self.peek().value == "not":
            self.consume()
            return Not(operand=self.parse_not())
        return self.parse_compare()

    def parse_compare(self) -> Node:
        left = self.parse_add()
        tok = self.peek()
        if tok.kind == "op" and tok.value in {">", ">=", "<", "<=", "==", "!="}:
            self.consume()
            right = self.parse_add()
            # op membership is guarded above; pydantic re-validates the Literal.
            return Compare(
                op=cast(Literal[">", ">=", "<", "<=", "==", "!="], tok.value),
                left=left,
                right=right,
            )
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.peek().kind == "op" and self.peek().value in {"+", "-"}:
            op = self.consume().value
            right = self.parse_mul()
            left = BinOp(
                op=cast(Literal["+", "-", "*", "/"], op), left=left, right=right
            )
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while self.peek().kind == "op" and self.peek().value in {"*", "/"}:
            op = self.consume().value
            right = self.parse_unary()
            left = BinOp(
                op=cast(Literal["+", "-", "*", "/"], op), left=left, right=right
            )
        return left

    def parse_unary(self) -> Node:
        if self.peek().kind == "op" and self.peek().value in {"+", "-"}:
            op = self.consume().value
            return UnaryOp(op=cast(Literal["-", "+"], op), operand=self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "num":
            self.consume()
            return Num(value=float(tok.value))
        if tok.kind == "lp":
            self.consume()
            node = self.parse_or()
            self.expect("rp")
            return node
        if tok.kind == "name":
            self.consume()
            if self.peek().kind == "lp":
                self.consume()
                args: list[Node] = []
                if self.peek().kind != "rp":
                    args.append(self.parse_or())
                    while self.peek().kind == "comma":
                        self.consume()
                        args.append(self.parse_or())
                self.expect("rp")
                return Call(name=tok.value, args=args, col=tok.col)
            if tok.value in {"true", "false"}:
                return Num(value=1.0 if tok.value == "true" else 0.0)
            return Name(name=tok.value)
        raise PineSyntaxError(f"Unexpected token {tok.value!r} at column {tok.col}")


def parse(expr: str) -> Node:
    if not expr or not expr.strip():
        raise PineSyntaxError("Empty expression")
    return _Parser(_tokenize(expr)).parse()


# ── evaluator ────────────────────────────────────────────────────────


#: A resolved value inside the evaluator: a Series when evaluating one ticker's
#: bars, a column-per-ticker DataFrame when evaluating a :class:`PanelBars`.
#: Every operator the AST uses (``+ - * / > >= < <= == != & | ~``, ``.shift``,
#: ``.rolling``, ``.ewm``, ``.fillna``, ``.astype``) has identical column-wise
#: semantics on both, which is what lets one ``_eval`` serve both modes.
Vector = Union[pd.Series, pd.DataFrame]


class PanelBars:
    """Column-per-ticker view over several bar frames that share one index.

    Quacks like the bars ``DataFrame`` that :func:`_eval` expects — ``columns``,
    ``index``, ``empty`` and ``__getitem__`` — except each ``__getitem__``
    returns a *wide* frame (rows = bars, columns = tickers) instead of a Series.

    Built only from tickers whose bar indexes and column sets are already
    identical (see :func:`_panel_groups`), so stacking them never reindexes and
    never pads with NaN: column ``t`` of every intermediate equals the Series
    the single-ticker path would have produced for ``t``.
    """

    __slots__ = ("_frames", "columns", "index", "tickers")

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        index: pd.Index,
        tickers: tuple[str, ...],
        columns: frozenset[str],
    ) -> None:
        self._frames = frames
        self.index = index
        self.tickers = tickers
        self.columns = columns

    @property
    def empty(self) -> bool:
        return len(self.index) == 0 or not self.tickers

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self._frames[name]


def _as_float(obj: Vector) -> Vector:
    """``astype(float)``, skipped when the values are already ``float64``.

    ``astype`` on an already-float column is a full copy plus a trip through
    pandas' dtype machinery, and the AST hits it once per ``Name`` reference —
    the dominant share of the profile's ``astype`` calls. Nothing in the
    evaluator mutates its inputs, so reusing the object is safe (see
    ``test_evaluate_does_not_mutate_or_alias_into_caller_state``).
    """
    if isinstance(obj, pd.DataFrame):
        if all(dt == np.float64 for dt in obj.dtypes):
            return obj
        return obj.astype(float)
    if obj.dtype == np.float64:
        return obj
    return obj.astype(float)


def _to_numeric(obj: Vector) -> Vector:
    """Coerce non-price columns to float, turning unparseable values into NaN."""
    if isinstance(obj, pd.DataFrame):
        if all(dt == np.float64 for dt in obj.dtypes):
            return obj
        coerced: pd.DataFrame = obj.apply(pd.to_numeric, errors="coerce")
        return coerced.astype(float)
    return _as_float(pd.to_numeric(obj, errors="coerce"))


def _broadcast(value, bars: pd.DataFrame | PanelBars) -> Vector:
    """Lift a Python scalar to the shape ``bars`` evaluates to."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value
    if isinstance(bars, PanelBars):
        return pd.DataFrame(
            float(value), index=bars.index, columns=list(bars.tickers), dtype=float
        )
    return pd.Series(float(value), index=bars.index)


def _as_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _require_int_literal(node: Node, func: str, arg: str) -> int:
    if not isinstance(node, Num):
        raise PineSyntaxError(f"{func}() argument {arg!r} must be an integer literal")
    n = int(node.value)
    if n != node.value or n <= 0:
        raise PineSyntaxError(
            f"{func}() argument {arg!r} must be a positive integer, got {node.value}"
        )
    return n


def _rsi(source: Vector, length: int) -> Vector:
    return wilder_rsi(source, length, min_periods=length)


def _atr(bars: pd.DataFrame | PanelBars, length: int) -> Vector:
    # wilder_atr/true_range are Series/DataFrame-overloaded, so the panel case
    # needs no special handling beyond the wide frames __getitem__ returns.
    if isinstance(bars, PanelBars):
        return wilder_atr(
            cast(pd.DataFrame, _as_float(bars["high"])),
            cast(pd.DataFrame, _as_float(bars["low"])),
            cast(pd.DataFrame, _as_float(bars["close"])),
            length,
            min_periods=length,
        )
    return wilder_atr(
        cast(pd.Series, _as_float(bars["high"])),
        cast(pd.Series, _as_float(bars["low"])),
        cast(pd.Series, _as_float(bars["close"])),
        length,
        min_periods=length,
    )


def _crossover(a: Vector, b: Vector) -> Vector:
    a_now = a
    b_now = b
    a_prev = a.shift(1)
    b_prev = b.shift(1)
    cond = (a_now > b_now) & (a_prev <= b_prev)
    return cond.fillna(False)


def _crossunder(a: Vector, b: Vector) -> Vector:
    a_now = a
    b_now = b
    a_prev = a.shift(1)
    b_prev = b.shift(1)
    cond = (a_now < b_now) & (a_prev >= b_prev)
    return cond.fillna(False)


def _series_from_name(name: str, bars: pd.DataFrame | PanelBars) -> Vector:
    if name == "adj_close":
        # with auto_adjust=True, close IS adjusted; alias them
        if "adj_close" in bars.columns:
            return _as_float(bars["adj_close"])
        return _as_float(bars["close"])
    if name in SERIES_NAMES:
        if name not in bars.columns:
            raise PineNameError(f"Series {name!r} not available in bars DataFrame")
        return _as_float(bars[name])
    if name in bars.columns:
        return _to_numeric(bars[name])
    raise PineNameError(f"Unknown identifier: {name!r}")


def _node_key(node: Node) -> tuple:
    """Return a location-independent structural key for an expression node.

    Parser column offsets are deliberately excluded: two identical subexpressions
    at different source locations have identical evaluation semantics and should
    share their computed vector.
    """
    if isinstance(node, Num):
        return ("num", float(node.value))
    if isinstance(node, Name):
        return ("name", node.name)
    if isinstance(node, UnaryOp):
        return ("unary", node.op, _node_key(node.operand))
    if isinstance(node, Not):
        return ("not", _node_key(node.operand))
    if isinstance(node, BinOp):
        return ("bin", node.op, _node_key(node.left), _node_key(node.right))
    if isinstance(node, Compare):
        return ("compare", node.op, _node_key(node.left), _node_key(node.right))
    if isinstance(node, BoolOp):
        return ("bool", node.op, _node_key(node.left), _node_key(node.right))
    if isinstance(node, Call):
        return ("call", node.name, tuple(_node_key(arg) for arg in node.args))
    raise PineSyntaxError(  # pragma: no cover - all Node variants handled above
        f"Unknown AST node: {type(node).__name__}"
    )


def _eval(
    node: Node,
    bars: pd.DataFrame | PanelBars,
    cache: dict[tuple, object] | None = None,
):
    """Walk the AST once against ``bars``.

    ``bars`` is either a single ticker's OHLCV frame (intermediates are Series)
    or a :class:`PanelBars` (intermediates are column-per-ticker DataFrames).
    The operators below are column-wise on DataFrames, so the panel case walks
    the identical code and pays one pandas call per node for *all* tickers.

    Structurally identical subexpressions share their vector within one
    evaluation. This matters for strategy expressions such as Minervini's,
    where the same SMA is referenced several times.
    """
    if cache is None:
        cache = {}
    key = _node_key(node)
    cached = cache.get(key)
    if cached is not None:
        return cached

    result: object
    if isinstance(node, Num):
        result = float(node.value)
    elif isinstance(node, Name):
        result = _series_from_name(node.name, bars)
    elif isinstance(node, UnaryOp):
        val = _eval(node.operand, bars, cache)
        if node.op == "-":
            result = -val
        else:
            result = val
    elif isinstance(node, Not):
        val = _broadcast(_eval(node.operand, bars, cache), bars).astype(bool)
        result = ~val
    elif isinstance(node, BinOp):
        left = _eval(node.left, bars, cache)
        right = _eval(node.right, bars, cache)
        if node.op == "+":
            result = left + right
        elif node.op == "-":
            result = left - right
        elif node.op == "*":
            result = left * right
        elif node.op == "/":
            result = left / right
        else:
            raise PineSyntaxError(  # pragma: no cover - op is a validated Literal
                f"Unknown operator: {node.op!r}"
            )
    elif isinstance(node, Compare):
        left = _eval(node.left, bars, cache)
        right = _eval(node.right, bars, cache)
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        out = ops[node.op](left, right)
        if isinstance(out, (pd.Series, pd.DataFrame)):
            result = out.fillna(False)
        else:
            result = bool(out)
    elif isinstance(node, BoolOp):
        left = _broadcast(_eval(node.left, bars, cache), bars).astype(bool)
        right = _broadcast(_eval(node.right, bars, cache), bars).astype(bool)
        if node.op == "and":
            result = left & right
        else:
            result = left | right
    elif isinstance(node, Call):
        result = _eval_call(node, bars, cache)
    else:
        raise PineSyntaxError(  # pragma: no cover - all Node variants handled above
            f"Unknown AST node: {type(node).__name__}"
        )
    cache[key] = result
    return result


def _eval_call(
    node: Call,
    bars: pd.DataFrame | PanelBars,
    cache: dict[tuple, object],
):
    name = node.name
    if name not in FUNC_NAMES:
        raise PineNameError(f"Unknown function: {name!r} at column {node.col}")
    if name in ROLLING_FUNCS:
        if len(node.args) != 2:
            raise PineSyntaxError(
                f"{name}() takes 2 arguments (source, length), got {len(node.args)}"
            )
        source_val = _broadcast(_eval(node.args[0], bars, cache), bars)
        length = _require_int_literal(node.args[1], name, "length")
        if name == "sma":
            return source_val.rolling(length, min_periods=length).mean()
        if name == "ema":
            return source_val.ewm(span=length, adjust=False, min_periods=length).mean()
        if name == "rsi":
            return _rsi(source_val, length)
        if name == "highest":
            return source_val.rolling(length, min_periods=length).max()
        if name == "lowest":
            return source_val.rolling(length, min_periods=length).min()
    if name == "atr":
        if len(node.args) != 1:
            raise PineSyntaxError(
                f"atr() takes 1 argument (length), got {len(node.args)}"
            )
        length = _require_int_literal(node.args[0], "atr", "length")
        return _atr(bars, length)
    if name in {"crossover", "crossunder"}:
        if len(node.args) != 2:
            raise PineSyntaxError(f"{name}() takes 2 arguments, got {len(node.args)}")
        a = _broadcast(_eval(node.args[0], bars, cache), bars)
        b = _broadcast(_eval(node.args[1], bars, cache), bars)
        return _crossover(a, b) if name == "crossover" else _crossunder(a, b)
    raise PineNameError(  # pragma: no cover - name already validated in FUNC_NAMES
        f"Unknown function: {name!r}"
    )


def evaluate(node: Node, bars: pd.DataFrame) -> pd.Series:
    """Evaluate ``node`` against ``bars`` and return a Series aligned to bars.index.

    Booleans are returned as a bool Series; numeric results are returned as a
    float Series. Pure-scalar results are broadcast across ``bars.index``.
    """
    if bars.empty:
        return pd.Series(dtype=float)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise PineError(f"bars DataFrame missing columns: {sorted(missing)}")
    result = _eval(node, bars)
    if isinstance(result, pd.Series):
        return result
    return _as_series(result, bars.index)


def evaluate_many(nodes: Iterable[Node], bars: pd.DataFrame) -> list[pd.Series]:
    """Evaluate several expressions against one frame with a shared AST cache."""
    nodes = list(nodes)
    if bars.empty:
        return [pd.Series(dtype=float) for _node in nodes]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise PineError(f"bars DataFrame missing columns: {sorted(missing)}")
    cache: dict[tuple, object] = {}
    out: list[pd.Series] = []
    for node in nodes:
        result = _eval(node, bars, cache)
        out.append(
            result if isinstance(result, pd.Series) else _as_series(result, bars.index)
        )
    return out


_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _call_names(node: Node) -> set[str]:
    """Return every function name invoked in the expression."""
    found: set[str] = set()

    def visit(n: Node) -> None:
        if isinstance(n, Call):
            found.add(n.name)
            for arg in n.args:
                visit(arg)
        elif isinstance(n, (BinOp, Compare, BoolOp)):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, (UnaryOp, Not)):
            visit(n.operand)

    visit(node)
    return found


def _panel_column_names(node: Node) -> set[str]:
    """Columns the panel must materialise to evaluate ``node``.

    Only the referenced ones: an unused column never has to be stacked, and —
    more importantly — never splits a group. Fundamentals and options joins add
    columns to just the tickers they cover, so keying groups on the *full*
    column set would shatter the panel on exactly the strategies that need it.
    """
    names = set(collect_names(node))
    if "adj_close" in names:
        # _series_from_name falls back to close when adj_close is absent.
        names.add("close")
    if "atr" in _call_names(node):
        names |= {"high", "low", "close"}
    return names


def _is_naive_numpy_datetime_index(index: pd.Index) -> bool:
    """True for tz-naive DatetimeIndex backed by any numpy ``datetime64[unit]``.

    Pandas 3 often builds daily calendars as ``datetime64[us]`` (2.x used
    ``datetime64[ns]``). The fast path must accept every resolution; tz-aware
    indexes use a non-numpy ``DatetimeTZDtype`` and stay on the slow path so
    label identity (timezone) is preserved.
    """
    return (
        isinstance(index, pd.DatetimeIndex)
        and isinstance(index.dtype, np.dtype)
        and bool(np.issubdtype(index.dtype, np.datetime64))
    )


def panel_index_key(index: pd.Index) -> tuple[str, object] | None:
    """Identity of ``index`` for stacking frames positionally, or None if unsafe.

    Two frames may be stacked into one panel only when this key matches: their
    indexes are then element-wise equal *and* interchangeable as labels, so the
    stack is a pure relabelling rather than an alignment.

    The dtype is part of the identity, not decoration: every member of a group is
    handed the first member's index back. Timestamps compare and hash by instant,
    so tz-aware indexes over the same moments in different zones (or the same
    instants at a different resolution) would otherwise group and silently return
    results relabelled into the first member's timezone.

    Shared with ``core._precompute_filter_signals`` so the two panel paths cannot
    drift on this rule.
    """
    if not index.is_unique:
        # Stacking would align on duplicate labels instead of stacking positionally.
        return None
    if _is_naive_numpy_datetime_index(index):
        # Byte-hash the raw int64 ticks. Unit is already in str(dtype), so us/ns
        # indexes with the same instants stay in different groups (correct).
        # Falling through to tuple(index) boxes every Timestamp and is ~50x
        # slower - the pandas-3 default of datetime64[us] used to force that.
        index_key: object = index.to_numpy().view("i8").tobytes()
    else:
        try:
            index_key = tuple(index)
        except TypeError:  # pragma: no cover - exotic unhashable index
            return None
    return (str(index.dtype), index_key)


def _group_key(bars: pd.DataFrame, names: Iterable[str]) -> tuple | None:
    """Identity of the panel ``bars`` may join, or None if it must go alone.

    Two frames share a key only when their bar indexes are element-wise equal
    and they agree on which referenced columns exist — the two things that make
    stacking them a pure relabelling rather than an alignment.
    """
    index_key = panel_index_key(bars.index)
    if index_key is None:
        return None
    columns = frozenset(bars.columns)
    if not _REQUIRED_COLUMNS <= columns:
        # evaluate() raises for these; let it do so per ticker.
        return None
    return (*index_key, frozenset(n for n in names if n in columns))


def _build_panel(
    tickers: tuple[str, ...],
    frames: list[pd.DataFrame],
    names: Iterable[str],
) -> PanelBars:
    index = frames[0].index
    columns = frozenset(frames[0].columns)
    labels = list(tickers)
    wide: dict[str, pd.DataFrame] = {}
    for name in names:
        if name not in columns:
            continue
        # Stack through NumPy rather than pd.concat: the indexes are known equal
        # (that is the group invariant), so there is nothing to align, and this
        # skips concat's per-Series bookkeeping to land one contiguous float
        # block. Dtype is normalised once per name for the whole group instead
        # of once per ticker per AST reference.
        if name in SERIES_NAMES:
            columns_1d = [frame[name].to_numpy(dtype=float) for frame in frames]
        else:
            columns_1d = [
                pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
                for frame in frames
            ]
        block = np.empty((len(index), len(labels)), dtype=float)
        for position, values in enumerate(columns_1d):
            block[:, position] = values
        wide[name] = pd.DataFrame(block, index=index, columns=labels, copy=False)
    return PanelBars(wide, index, tickers, columns)


def _series_from_panel_column(
    values: np.ndarray, index: pd.Index, position: int
) -> pd.Series:
    """Build a Series for one panel column without ``DataFrame.iloc`` boxing."""
    return pd.Series(values[:, position], index=index, copy=False)


def _bool_array_from_series(series: pd.Series) -> np.ndarray:
    """Match ``series.fillna(False).astype(bool).to_numpy()`` with less work."""
    if series.dtype == bool:
        if not series.hasnans:
            return series.to_numpy(dtype=bool, copy=False)
        return series.fillna(False).to_numpy(dtype=bool)
    return series.fillna(False).astype(bool).to_numpy(dtype=bool)


def _bool_block_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Booleanize a panel result once for every ticker column."""
    if len(frame.columns) == 0:
        return np.empty((len(frame.index), 0), dtype=bool)
    # Fast path: already a bool block with no NA.
    if all(dt is bool or dt == np.dtype(bool) for dt in frame.dtypes):
        if not frame.isna().any().any():
            return frame.to_numpy(dtype=bool, copy=False)
        return frame.fillna(False).to_numpy(dtype=bool)
    return frame.fillna(False).astype(bool).to_numpy(dtype=bool)


def evaluate_panel(
    node: Node,
    bars_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.Series | PineError]:
    """Evaluate ``node`` for many tickers, batching aligned ones into panels.

    Returns one entry per ticker with non-empty bars: either the Series
    :func:`evaluate` would have returned, or the :class:`PineError` it would
    have raised. Results are identical to calling :func:`evaluate` per ticker -
    tickers are only stacked when their indexes match exactly, so no value ever
    sees a reindex-padded neighbour.
    """
    return evaluate_panel_many((node,), bars_by_ticker)[0]


def evaluate_panel_many(
    nodes: Iterable[Node],
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    as_bool_array: bool = False,
) -> list[dict[str, pd.Series | np.ndarray | PineError]]:
    """Evaluate multiple expressions over one set of compatible ticker panels.

    Panel construction and structurally identical subexpressions are shared
    across the expressions. Results retain the same per-expression/per-ticker
    error isolation as repeated :func:`evaluate_panel` calls.

    When ``as_bool_array`` is True, successful results are ``bool`` ndarrays
    aligned to each ticker's bar index instead of Series. Callers that only
    need matrices (entry signals feeding candidate matrices) skip the large
    per-ticker Series boxing cost that otherwise dominates the pine bucket.
    """
    nodes = list(nodes)
    if not nodes:
        return []
    names: set[str] = set()
    for node in nodes:
        names.update(_panel_column_names(node))
    groups: dict[tuple, list[str]] = {}
    solo: list[str] = []
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            continue
        key = _group_key(bars, names)
        if key is None:
            solo.append(ticker)
        else:
            groups.setdefault(key, []).append(ticker)

    outputs: list[dict[str, pd.Series | np.ndarray | PineError]] = [
        {} for _node in nodes
    ]

    def _store_series(position: int, ticker: str, series: pd.Series) -> None:
        if as_bool_array:
            outputs[position][ticker] = _bool_array_from_series(series)
        else:
            outputs[position][ticker] = series

    def eval_one(ticker: str) -> None:
        try:
            results = evaluate_many(nodes, bars_by_ticker[ticker])
        except PineError:
            # Find the exact failing expression/message without hiding results
            # from independent expressions.
            for position, node in enumerate(nodes):
                try:
                    _store_series(
                        position, ticker, evaluate(node, bars_by_ticker[ticker])
                    )
                except PineError as node_exc:
                    outputs[position][ticker] = node_exc
        else:
            for position, result in enumerate(results):
                _store_series(position, ticker, result)

    for ticker in solo:
        eval_one(ticker)

    for members in groups.values():
        if len(members) == 1:
            eval_one(members[0])
            continue
        tickers = tuple(members)
        frames = [bars_by_ticker[t] for t in tickers]
        try:
            panel = _build_panel(tickers, frames, names)
        except (PineError, TypeError, ValueError):
            for ticker in tickers:
                eval_one(ticker)
            continue
        cache: dict[tuple, object] = {}
        for node_position, node in enumerate(nodes):
            try:
                result = _broadcast(_eval(node, panel, cache), panel)
            except (PineError, TypeError, ValueError):
                # Redo only the failing expression one ticker at a time so its
                # exact exception is preserved without discarding other panel
                # results.
                for ticker in tickers:
                    try:
                        _store_series(
                            node_position,
                            ticker,
                            evaluate(node, bars_by_ticker[ticker]),
                        )
                    except PineError as exc:
                        outputs[node_position][ticker] = exc
                continue
            frame = cast(pd.DataFrame, result)
            if as_bool_array:
                bool_block = _bool_block_from_frame(frame)
                for ticker_position, ticker in enumerate(tickers):
                    # Copy so callers can retain columns independently of the
                    # temporary panel block.
                    outputs[node_position][ticker] = np.array(
                        bool_block[:, ticker_position], copy=True, dtype=bool
                    )
            else:
                values = frame.to_numpy(copy=False)
                index = frame.index
                for ticker_position, ticker in enumerate(tickers):
                    outputs[node_position][ticker] = _series_from_panel_column(
                        values, index, ticker_position
                    )
    return outputs


def required_lookback(node: Node) -> int:
    """Return the maximum rolling window length referenced in the expression.

    Used by the engine to verify each ticker has enough history prior to the
    signal bar. Functions like crossover/crossunder add 1 bar of lookback.
    """
    max_len = 0

    def visit(n: Node) -> None:
        nonlocal max_len
        if isinstance(n, Call):
            if (
                n.name in ROLLING_FUNCS
                and len(n.args) == 2
                and isinstance(n.args[1], Num)
            ):
                max_len = max(max_len, int(n.args[1].value))
            if n.name == "atr" and len(n.args) == 1 and isinstance(n.args[0], Num):
                max_len = max(max_len, int(n.args[0].value))
            if n.name in {"crossover", "crossunder"}:
                max_len = max(max_len, 1)
            for arg in n.args:
                visit(arg)
        elif isinstance(n, (BinOp, Compare, BoolOp)):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, (UnaryOp, Not)):
            visit(n.operand)

    visit(node)
    return max_len


def collect_names(node: Node) -> set[str]:
    """Return every bare identifier (``Name``) referenced in the expression.

    Function names (``Call.name``) are excluded — only series/column identifiers
    like ``close`` or ``revenue_up_3q`` are collected. Callers use this to decide
    which non-price columns (e.g. dated fundamentals) an expression depends on.
    """
    names: set[str] = set()

    def visit(n: Node) -> None:
        if isinstance(n, Name):
            names.add(n.name)
        elif isinstance(n, Call):
            for arg in n.args:
                visit(arg)
        elif isinstance(n, (BinOp, Compare, BoolOp)):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, (UnaryOp, Not)):
            visit(n.operand)

    visit(node)
    return names
