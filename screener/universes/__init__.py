from __future__ import annotations

from pathlib import Path
from typing import Optional

_UNIVERSES_DIR = Path(__file__).parent

_DEFAULTS: dict[str, str] = {
    "us": "sp500.txt",
    "india": "nifty500.txt",
}


def _read_file(path: Path) -> list[str]:
    tickers: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def load_universe(market: str, path: Optional[str] = None) -> list[str]:
    """Return ticker list for the given market.

    If *path* is given, read from that file (plain tickers, one per line,
    ``#``-prefixed comments ignored).  Otherwise use the bundled default for
    *market*.

    Note: the bundled S&P 500 / Nifty 500 lists reflect today's index
    membership and therefore carry survivorship bias.  For point-in-time
    accuracy supply a custom ``--universe`` file.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Universe file not found: {path}")
        return _read_file(p)

    fname = _DEFAULTS.get(market)
    if fname is None:
        raise ValueError(
            f"No default universe for market '{market}'. "
            "Provide --universe <path> with a list of tickers."
        )
    return _read_file(_UNIVERSES_DIR / fname)
