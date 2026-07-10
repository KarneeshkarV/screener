from __future__ import annotations

import pytest

from screener.parallel import parallel_map


def test_parallel_map_collects_results_completion_order() -> None:
    out = parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2)

    assert sorted(out) == [2, 4, 6]


def test_parallel_map_drops_none_by_default() -> None:
    out = parallel_map(lambda x: x if x % 2 else None, [1, 2, 3], max_workers=2)

    assert sorted(out) == [1, 3]


def test_parallel_map_can_keep_none_results() -> None:
    out = parallel_map(
        lambda x: x if x % 2 else None,
        [1, 2, 3],
        max_workers=2,
        drop_none=False,
    )

    assert sorted(value for value in out if value is not None) == [1, 3]
    assert out.count(None) == 1


def test_parallel_map_propagates_errors_by_default() -> None:
    def fn(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        parallel_map(fn, [1, 2, 3], max_workers=2)


def test_parallel_map_skips_errors_when_requested() -> None:
    def fn(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    out = parallel_map(fn, [1, 2, 3], max_workers=2, on_error="skip")

    assert sorted(out) == [1, 3]


def test_parallel_map_max_workers_one_path() -> None:
    out = parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=1)

    assert out == [2, 3, 4]
