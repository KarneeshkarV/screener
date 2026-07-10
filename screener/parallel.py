"""Small shared helpers for bounded parallel fan-out."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, TypeVar, overload

T = TypeVar("T")
R = TypeVar("R")
OnError = Literal["raise", "skip"]


@overload
def parallel_map(
    fn: Callable[[T], R | None],
    items: Iterable[T],
    *,
    max_workers: int,
    drop_none: Literal[True] = True,
    on_error: OnError = "raise",
) -> list[R]: ...


@overload
def parallel_map(
    fn: Callable[[T], R | None],
    items: Iterable[T],
    *,
    max_workers: int,
    drop_none: Literal[False],
    on_error: OnError = "raise",
) -> list[R | None]: ...


def parallel_map(
    fn: Callable[[T], R | None],
    items: Iterable[T],
    *,
    max_workers: int,
    drop_none: bool = True,
    on_error: OnError = "raise",
) -> list[R] | list[R | None]:
    """Apply ``fn`` to ``items`` concurrently and collect completion-order results."""
    results: list[R | None] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                if on_error == "skip":
                    continue
                raise
            if result is None and drop_none:
                continue
            results.append(result)
    return results
