"""Direct unit tests for the shared reporting mechanics.

These pin the leaf behaviours that ``unusual_volume.output`` and
``rs_breakout`` now delegate to. The end-to-end byte-identical proof for the
JSON/Markdown *files* lives in ``tests/test_unusual_volume_detection_delivery.py`` and
``tests/test_rs_breakout.py`` (e.g. ``test_write_json_sanitizes_nonfinite_metrics``,
``test_write_markdown_buckets_and_headers``, ``test_write_json_serializes_result_dates``);
those still exercise the real writers through the delegating wrappers.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.reporting import dump_json_file, json_safe, markdown_row

# ─────────────────────────── json_safe ───────────────────────────


def test_json_safe_none_and_missing():
    assert json_safe(None) is None
    assert json_safe(float("nan")) is None
    assert json_safe(pd.NA) is None
    assert json_safe(pd.NaT) is None


def test_json_safe_non_finite_floats():
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None


def test_json_safe_bool_not_coerced():
    # bool is a Real subclass but must stay a bool, not become 1/0.
    assert json_safe(True) is True
    assert json_safe(False) is False


def test_json_safe_numpy_scalars():
    assert json_safe(np.int64(5)) == 5
    assert isinstance(json_safe(np.int64(5)), int)
    assert json_safe(np.float64(2.5)) == 2.5
    assert isinstance(json_safe(np.float64(2.5)), float)


def test_json_safe_recurses_containers():
    assert json_safe({"a": [float("nan"), pd.NA], "b": (1, 2)}) == {
        "a": [None, None],
        "b": [1, 2],  # tuple becomes list
    }


def test_json_safe_passes_through_arraylike_and_str():
    # pd.isna raises on a Series -> value flows through untouched.
    s = pd.Series([1, 2])
    assert json_safe(s) is s
    assert json_safe("hello") == "hello"


# ─────────────────────────── dump_json_file ───────────────────────────


def test_dump_json_file_exact_kwargs(tmp_path):
    path = tmp_path / "out.json"
    dump_json_file({"d": date(2026, 4, 30), "n": 1}, path)
    text = path.read_text()
    # indent=2 -> newlines + two-space indentation; default=str -> date string.
    assert text == '{\n  "d": "2026-04-30",\n  "n": 1\n}'
    # No trailing newline is appended.
    assert not text.endswith("\n")
    assert json.loads(text)["d"] == "2026-04-30"


def test_dump_json_file_allow_nan_default_true_writes_nan(tmp_path):
    path = tmp_path / "nan.json"
    dump_json_file({"x": float("nan")}, path)
    assert "NaN" in path.read_text()


def test_dump_json_file_allow_nan_false_raises(tmp_path):
    path = tmp_path / "strict.json"
    with pytest.raises(ValueError):
        dump_json_file({"x": float("nan")}, path, allow_nan=False)


def test_dump_json_file_accepts_str_path(tmp_path):
    path = tmp_path / "str.json"
    dump_json_file([1, 2], str(path))
    assert json.loads(path.read_text()) == [1, 2]


# ─────────────────────────── markdown_row ───────────────────────────


def test_markdown_row_shape():
    assert markdown_row(["a", "b", "c"]) == "| a | b | c |"


def test_markdown_row_single_cell():
    assert markdown_row(["x"]) == "| x |"
