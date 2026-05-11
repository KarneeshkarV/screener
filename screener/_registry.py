"""Tiny plugin-registry primitive shared by strategies, criteria, and indicators.

Each registry stores ``name -> value`` plus optional metadata. Plugin modules
register entries with the ``register`` decorator. ``autodiscover`` imports every
submodule of a package so module-level decorators fire on first import.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, ItemsView, Iterator
from types import ModuleType
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, label: str) -> None:
        self._label = label
        self._entries: dict[str, T] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def register(self, name: str, **meta: Any) -> Callable[[T], T]:
        """Decorator: ``@reg.register("foo", aliased=False)``."""

        def _wrap(value: T) -> T:
            self.add(name, value, **meta)
            return value

        return _wrap

    def add(self, name: str, value: T, **meta: Any) -> None:
        if name in self._entries:
            raise ValueError(f"{self._label} already has {name!r}")
        self._entries[name] = value
        if meta:
            self._meta[name] = dict(meta)

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self._label} {name!r}. Known: {sorted(self._entries)}"
            ) from None

    def get_optional(self, name: str | None) -> T | None:
        if name is None:
            return None
        return self._entries.get(name)

    def names(self) -> list[str]:
        return list(self._entries)

    def items(self) -> ItemsView[str, T]:
        return self._entries.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def meta(self, name: str) -> dict[str, Any]:
        return dict(self._meta.get(name, {}))

    def as_dict(self) -> dict[str, T]:
        """Return a snapshot dict — handy for backwards-compat exports."""
        return dict(self._entries)


def autodiscover(package: ModuleType) -> None:
    """Import every submodule of ``package`` so registration side effects fire."""
    if not hasattr(package, "__path__"):
        raise TypeError(f"autodiscover expects a package, got {package!r}")
    for mod_info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{mod_info.name}")
