import math
import re

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

SETUP_SCORE_COLUMNS = [
    "EMA5",
    "EMA20",
    "EMA100",
    "EMA200",
    "RSI",
]

DETAIL_COLUMNS = [
    "price_earnings_ttm",
    "return_on_equity",
    "dividend_yield_recent",
    "debt_to_equity",
    "RSI",
]


def _percentile(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True).fillna(0)


def _log_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return _percentile(values.add(1).map(math.log))


def _add_setup_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    close = pd.to_numeric(scored["close"], errors="coerce")
    ema5 = pd.to_numeric(scored["EMA5"], errors="coerce")
    ema20 = pd.to_numeric(scored["EMA20"], errors="coerce")
    ema100 = pd.to_numeric(scored["EMA100"], errors="coerce")
    ema200 = pd.to_numeric(scored["EMA200"], errors="coerce")
    change = pd.to_numeric(scored["change"], errors="coerce")
    rsi = pd.to_numeric(scored["RSI"], errors="coerce")

    dollar_volume = pd.to_numeric(scored["volume"], errors="coerce") * close
    liquidity = _log_percentile(dollar_volume)
    market_cap = _log_percentile(scored["market_cap_basic"])

    trend_spread = (
        ((ema5 - ema20) / close)
        + ((ema20 - ema100) / close)
        + ((ema100 - ema200) / close)
    ).clip(lower=0, upper=0.35)
    trend_strength = _percentile(trend_spread)

    momentum = ((change.clip(lower=-5, upper=10) + 5) / 15).fillna(0)
    rsi_quality = (1 - ((rsi - 60).abs() / 40)).clip(lower=0, upper=1).fillna(0)
    price_quality = _percentile(close.clip(lower=0, upper=200))

    extension = ((close - ema20) / ema20).fillna(0)
    overextension_penalty = ((extension - 0.12).clip(lower=0) / 0.25).clip(upper=1)

    scored["setup_score"] = (
        25 * liquidity
        + 30 * trend_strength
        + 15 * momentum
        + 15 * market_cap
        + 10 * rsi_quality
        + 5 * price_quality
        - 15 * overextension_penalty
    ).round(2)
    return scored


def _dedupe_listings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "description" not in df.columns:
        return df

    deduped = df.copy()
    fallback = deduped["name"] if "name" in deduped.columns else deduped["ticker"]
    company = deduped["description"].fillna("").where(
        deduped["description"].fillna("").str.strip() != "",
        fallback.fillna(""),
    )
    deduped["_listing_key"] = company.map(
        lambda value: re.sub(r"[^a-z0-9]+", "", str(value).lower())
    )
    deduped = deduped.drop_duplicates("_listing_key", keep="first")
    return deduped.drop(columns=["_listing_key"])


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

    if order_by == "setup_score":
        columns.extend(c for c in SETUP_SCORE_COLUMNS if c not in columns)
        fetch_limit = max(limit * 10, 500)
    else:
        fetch_limit = max(limit * 3, 100)

    query = (
        Query()
        .set_markets(MARKETS[market])
        .select(*columns)
        .where(*filters)
        .order_by("volume" if order_by == "setup_score" else order_by, ascending=False)
        .limit(fetch_limit)
    )

    count, df = query.get_scanner_data()
    if order_by == "setup_score" and not df.empty:
        df = _add_setup_score(df)
        df = df.sort_values("setup_score", ascending=False)
        hidden_score_columns = [
            col for col in SETUP_SCORE_COLUMNS if not detail or col not in DETAIL_COLUMNS
        ]
        df = df.drop(columns=hidden_score_columns)
    if not df.empty:
        df = _dedupe_listings(df).head(limit)

    return count, df
