from __future__ import annotations

from threading import Event, Lock

import pytest

from screener.parallel import parallel_map


def test_parallel_map_collects_results_in_input_order() -> None:
    out = parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2)

    assert out == [2, 4, 6]


def test_parallel_map_preserves_input_order_when_later_items_finish_first() -> None:
    others_done = Event()
    pending = {"count": 3}
    lock = Lock()

    def fn(x: int) -> int:
        if x == 0:
            others_done.wait(timeout=5)
            return 0
        with lock:
            pending["count"] -= 1
            if pending["count"] == 0:
                others_done.set()
        return x

    out = parallel_map(fn, [0, 1, 2, 3], max_workers=4)

    assert out == [0, 1, 2, 3]


def test_parallel_map_drops_none_by_default_preserving_order() -> None:
    out = parallel_map(lambda x: x if x % 2 else None, [1, 2, 3], max_workers=2)

    assert out == [1, 3]


def test_parallel_map_can_keep_none_results() -> None:
    out = parallel_map(
        lambda x: x if x % 2 else None,
        [1, 2, 3],
        max_workers=2,
        drop_none=False,
    )

    assert out == [1, None, 3]


def test_parallel_map_propagates_errors_by_default() -> None:
    def fn(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        parallel_map(fn, [1, 2, 3], max_workers=2)


def test_parallel_map_skips_errors_preserving_input_relative_order() -> None:
    def fn(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    out = parallel_map(fn, [1, 2, 3], max_workers=2, on_error="skip")

    assert out == [1, 3]


def test_parallel_map_max_workers_one_path() -> None:
    out = parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=1)

    assert out == [2, 3, 4]
