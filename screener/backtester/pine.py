"""Safe Pine-like expression parser and evaluator.

Parses a small subset of Pine Script into an AST and evaluates it against a
pandas DataFrame of OHLCV bars. Does NOT use Python ``eval``.

Supported:
  series: open, high, low, close, volume, adj_close
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

from dataclasses import dataclass

import pandas as pd

from screener import indicators


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


@dataclass
class Num:
    value: float


@dataclass
class Name:
    name: str


@dataclass
class UnaryOp:
    op: str            # '-' or '+'
    operand: Node


@dataclass
class Not:
    operand: Node


@dataclass
class BinOp:
    op: str            # '+', '-', '*', '/'
    left: Node
    right: Node


@dataclass
class Compare:
    op: str            # '>', '>=', '<', '<=', '==', '!='
    left: Node
    right: Node


@dataclass
class BoolOp:
    op: str            # 'and' or 'or'
    left: Node
    right: Node


@dataclass
class Call:
    name: str
    args: list
    col: int           # column in source (for error messages)


Node = Num | Name | UnaryOp | Not | BinOp | Compare | BoolOp | Call


# ── tokenizer ────────────────────────────────────────────────────────


_TWO_CHAR = {">=", "<=", "==", "!="}
_SINGLE_PUNCT = set("+-*/()><,")


@dataclass
class Token:
    kind: str          # 'num', 'name', 'op', 'lp', 'rp', 'comma', 'end'
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
            tokens.append(Token("num", expr[start:i], start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            tokens.append(Token("name", expr[start:i], start))
            continue
        two = expr[i:i + 2]
        if two in _TWO_CHAR:
            tokens.append(Token("op", two, i))
            i += 2
            continue
        if ch == "(":
            tokens.append(Token("lp", ch, i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("rp", ch, i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token("comma", ch, i))
            i += 1
            continue
        if ch in _SINGLE_PUNCT:
            tokens.append(Token("op", ch, i))
            i += 1
            continue
        if ch == "!":
            raise PineSyntaxError(
                f"Unexpected {ch!r} at column {i}; did you mean '!='?"
            )
        raise PineSyntaxError(f"Unexpected character {ch!r} at column {i}")
    tokens.append(Token("end", "", n))
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
            raise PineSyntaxError(
                f"Unexpected {tok.value!r} at column {tok.col}"
            )
        return node

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.peek().kind == "name" and self.peek().value == "or":
            self.consume()
            right = self.parse_and()
            left = BoolOp("or", left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while self.peek().kind == "name" and self.peek().value == "and":
            self.consume()
            right = self.parse_not()
            left = BoolOp("and", left, right)
        return left

    def parse_not(self) -> Node:
        if self.peek().kind == "name" and self.peek().value == "not":
            self.consume()
            return Not(self.parse_not())
        return self.parse_compare()

    def parse_compare(self) -> Node:
        left = self.parse_add()
        tok = self.peek()
        if tok.kind == "op" and tok.value in {">", ">=", "<", "<=", "==", "!="}:
            self.consume()
            right = self.parse_add()
            return Compare(tok.value, left, right)
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.peek().kind == "op" and self.peek().value in {"+", "-"}:
            op = self.consume().value
            right = self.parse_mul()
            left = BinOp(op, left, right)
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while self.peek().kind == "op" and self.peek().value in {"*", "/"}:
            op = self.consume().value
            right = self.parse_unary()
            left = BinOp(op, left, right)
        return left

    def parse_unary(self) -> Node:
        if self.peek().kind == "op" and self.peek().value in {"+", "-"}:
            op = self.consume().value
            return UnaryOp(op, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "num":
            self.consume()
            return Num(float(tok.value))
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
                return Call(tok.value, args, tok.col)
            if tok.value in {"true", "false"}:
                return Num(1.0 if tok.value == "true" else 0.0)
            return Name(tok.value)
        raise PineSyntaxError(
            f"Unexpected token {tok.value!r} at column {tok.col}"
        )


def parse(expr: str) -> Node:
    if not expr or not expr.strip():
        raise PineSyntaxError("Empty expression")
    return _Parser(_tokenize(expr)).parse()


# ── evaluator ────────────────────────────────────────────────────────


def _as_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _require_int_literal(node: Node, func: str, arg: str) -> int:
    if not isinstance(node, Num):
        raise PineSyntaxError(
            f"{func}() argument {arg!r} must be an integer literal"
        )
    n = int(node.value)
    if n != node.value or n <= 0:
        raise PineSyntaxError(
            f"{func}() argument {arg!r} must be a positive integer, got {node.value}"
        )
    return n


def _rsi(source: pd.Series, length: int) -> pd.Series:
    return indicators.rsi(source, length)


def _atr(bars: pd.DataFrame, length: int) -> pd.Series:
    return indicators.atr(bars["high"], bars["low"], bars["close"], length)


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return indicators.crossover(a, b)


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return indicators.crossunder(a, b)


def _series_from_name(name: str, bars: pd.DataFrame) -> pd.Series:
    if name == "adj_close":
        # with auto_adjust=True, close IS adjusted; alias them
        if "adj_close" in bars.columns:
            return bars["adj_close"].astype(float)
        return bars["close"].astype(float)
    if name in SERIES_NAMES:
        if name not in bars.columns:
            raise PineNameError(
                f"Series {name!r} not available in bars DataFrame"
            )
        return bars[name].astype(float)
    raise PineNameError(f"Unknown identifier: {name!r}")


def _eval(node: Node, bars: pd.DataFrame):
    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Name):
        return _series_from_name(node.name, bars)
    if isinstance(node, UnaryOp):
        val = _eval(node.operand, bars)
        if node.op == "-":
            return -val
        return val
    if isinstance(node, Not):
        val = _as_series(_eval(node.operand, bars), bars.index).astype(bool)
        return ~val
    if isinstance(node, BinOp):
        left = _eval(node.left, bars)
        right = _eval(node.right, bars)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        raise PineSyntaxError(f"Unknown operator: {node.op!r}")
    if isinstance(node, Compare):
        left = _eval(node.left, bars)
        right = _eval(node.right, bars)
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        out = ops[node.op](left, right)
        if isinstance(out, pd.Series):
            return out.fillna(False)
        return bool(out)
    if isinstance(node, BoolOp):
        left = _as_series(_eval(node.left, bars), bars.index).astype(bool)
        right = _as_series(_eval(node.right, bars), bars.index).astype(bool)
        if node.op == "and":
            return left & right
        return left | right
    if isinstance(node, Call):
        return _eval_call(node, bars)
    raise PineSyntaxError(f"Unknown AST node: {type(node).__name__}")


def _eval_call(node: Call, bars: pd.DataFrame):
    name = node.name
    if name not in FUNC_NAMES:
        raise PineNameError(f"Unknown function: {name!r} at column {node.col}")
    if name in ROLLING_FUNCS:
        if len(node.args) != 2:
            raise PineSyntaxError(
                f"{name}() takes 2 arguments (source, length), got {len(node.args)}"
            )
        source_val = _eval(node.args[0], bars)
        if not isinstance(source_val, pd.Series):
            source_val = _as_series(source_val, bars.index)
        length = _require_int_literal(node.args[1], name, "length")
        if name == "sma":
            return indicators.sma(source_val, length)
        if name == "ema":
            return indicators.ema(source_val, length)
        if name == "rsi":
            return indicators.rsi(source_val, length)
        if name == "highest":
            return indicators.highest(source_val, length)
        if name == "lowest":
            return indicators.lowest(source_val, length)
    if name == "atr":
        if len(node.args) != 1:
            raise PineSyntaxError(
                f"atr() takes 1 argument (length), got {len(node.args)}"
            )
        length = _require_int_literal(node.args[0], "atr", "length")
        return _atr(bars, length)
    if name in {"crossover", "crossunder"}:
        if len(node.args) != 2:
            raise PineSyntaxError(
                f"{name}() takes 2 arguments, got {len(node.args)}"
            )
        a = _as_series(_eval(node.args[0], bars), bars.index)
        b = _as_series(_eval(node.args[1], bars), bars.index)
        return _crossover(a, b) if name == "crossover" else _crossunder(a, b)
    raise PineNameError(f"Unknown function: {name!r}")


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


def required_lookback(node: Node) -> int:
    """Return the maximum rolling window length referenced in the expression.

    Used by the engine to verify each ticker has enough history prior to the
    signal bar. Functions like crossover/crossunder add 1 bar of lookback.
    """
    max_len = 0

    def visit(n: Node) -> None:
        nonlocal max_len
        if isinstance(n, Call):
            if n.name in ROLLING_FUNCS and len(n.args) == 2 and isinstance(n.args[1], Num):
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
