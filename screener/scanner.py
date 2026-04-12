from tradingview_screener import Query
import pandas as pd


MARKETS = {
    "us": "america",
    "india": "india",
}

DEFAULT_COLUMNS = [
    "name",
    "description",
    "close",
    "change",
    "volume",
    "market_cap_basic",
]

DETAIL_COLUMNS = [
    "price_earnings_ttm",
    "return_on_equity",
    "dividend_yield_recent",
    "debt_to_equity",
    "RSI",
]


def scan(
    market: str,
    filters: list,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
) -> tuple[int, pd.DataFrame]:
    columns = list(DEFAULT_COLUMNS)
    if detail:
        columns.extend(DETAIL_COLUMNS)

    query = (
        Query()
        .set_markets(MARKETS[market])
        .select(*columns)
        .where(*filters)
        .order_by(order_by, ascending=False)
        .limit(limit)
    )

    count, df = query.get_scanner_data()
    return count, df
