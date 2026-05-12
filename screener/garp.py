"""GARP screen helpers for India and US markets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from screener.cache import cached_json_call
from screener.scanner import scan


INDIA_MIN_CRORE = 1000.0
US_MIN_USD = 1_000_000_000.0


class GarpThresholds(BaseModel):
    market_cap_min: float = Field(ge=0.0)
    sales_min: float = Field(ge=0.0)
    peg_max: float = Field(default=2.0, gt=0.0)
    sales_growth_5y_min: float = Field(default=15.0)
    operating_profit_growth_min: float = Field(default=10.0)
    eps_growth_5y_min: float = Field(default=12.0)
    roe_5y_min: float = Field(default=15.0)
    roce_or_roic_min: float = Field(default=15.0)

    model_config = ConfigDict(frozen=True)


INDIA_THRESHOLDS = GarpThresholds(
    market_cap_min=INDIA_MIN_CRORE,
    sales_min=INDIA_MIN_CRORE,
)
US_THRESHOLDS = GarpThresholds(market_cap_min=US_MIN_USD, sales_min=US_MIN_USD)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _first_num(mapping: dict[str, Any], *keys: str) -> float | None:
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower())
        parsed = _num(value)
        if parsed is not None:
            return parsed
    return None


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return ((new - old) / abs(old)) * 100.0


def _cagr(latest: float | None, oldest: float | None, years: float) -> float | None:
    if latest is None or oldest is None or latest <= 0 or oldest <= 0 or years <= 0:
        return None
    return ((latest / oldest) ** (1.0 / years) - 1.0) * 100.0


def _series_from_statement(statement: pd.DataFrame, row_names: list[str]) -> pd.Series:
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    for name in row_names:
        if name in statement.index:
            return pd.to_numeric(statement.loc[name], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _average_ratio(
    numerator: pd.Series, denominator: pd.Series, periods: int
) -> float | None:
    if numerator.empty or denominator.empty:
        return None
    values = []
    for col in list(numerator.index)[:periods]:
        den = _num(denominator.get(col))
        num = _num(numerator.get(col))
        if num is not None and den not in (None, 0):
            values.append((num / den) * 100.0)
    if not values:
        return None
    return float(sum(values) / len(values))


def _passes_garp(row: dict[str, Any], thresholds: GarpThresholds) -> bool:
    required = [
        row.get("market_cap"),
        row.get("sales"),
        row.get("peg"),
        row.get("sales_growth_5y"),
        row.get("operating_profit_growth"),
        row.get("eps_growth_5y"),
        row.get("roe_5y"),
        row.get("roce_or_roic"),
        row.get("quarterly_profit_growth"),
    ]
    if any(_num(value) is None for value in required):
        return False
    return (
        float(row["market_cap"]) > thresholds.market_cap_min
        and float(row["sales"]) > thresholds.sales_min
        and 0 < float(row["peg"]) < thresholds.peg_max
        and float(row["sales_growth_5y"]) > thresholds.sales_growth_5y_min
        and float(row["operating_profit_growth"])
        > thresholds.operating_profit_growth_min
        and float(row["eps_growth_5y"]) > thresholds.eps_growth_5y_min
        and float(row["roe_5y"]) > thresholds.roe_5y_min
        and float(row["roce_or_roic"]) > thresholds.roce_or_roic_min
        and float(row["quarterly_profit_growth"]) > 0
    )


def add_garp_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    if scored.empty:
        scored["garp_score"] = []
        return scored

    def pct(col: str) -> pd.Series:
        return pd.to_numeric(scored[col], errors="coerce").rank(pct=True).fillna(0)

    peg = pd.to_numeric(scored["peg"], errors="coerce")
    inv_peg = (1 - peg.rank(pct=True)).fillna(0)
    scored["garp_score"] = (
        30 * inv_peg
        + 20 * pct("eps_growth_5y")
        + 15 * pct("sales_growth_5y")
        + 15 * pct("roe_5y")
        + 10 * pct("roce_or_roic")
        + 10 * pct("quarterly_profit_growth")
    ).round(2)
    return scored.sort_values("garp_score", ascending=False)


def load_garp_universe(
    market: str,
    universe_size: int,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    from tradingview_screener import col

    if market == "india":
        filters = [
            col("type") == "stock",
            col("close") >= 10,
            col("market_cap_basic") >= INDIA_MIN_CRORE,
        ]
    else:
        filters = [
            col("type") == "stock",
            col("close") >= 1,
            col("market_cap_basic") >= US_MIN_USD,
        ]
    _total, df = scan(
        market=market,
        filters=filters,
        limit=universe_size,
        order_by="volume",
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return df


def _fetch_india_sections(symbol: str) -> dict[str, Any]:
    from openscreener import Stock

    stock = Stock(symbol)
    return {
        "ratios": stock.fetch("ratios") or {},
        "profit_loss": stock.fetch("profit_loss") or {},
        "quarterly_results": stock.fetch("quarterly_results") or {},
    }


def _india_row(
    symbol: str, description: str | None, payload: dict[str, Any]
) -> dict[str, Any]:
    ratios = cast(
        dict[str, Any],
        payload.get("ratios") if isinstance(payload.get("ratios"), dict) else {},
    )
    profit_loss = cast(
        dict[str, Any],
        payload.get("profit_loss")
        if isinstance(payload.get("profit_loss"), dict)
        else {},
    )
    metrics = {**profit_loss, **ratios}
    quarterly = (
        payload.get("quarterly_results")
        if isinstance(payload.get("quarterly_results"), dict)
        else {}
    )
    expected_q_np = _first_num(
        ratios,
        "expected_quarterly_net_profit",
        "expected_quarterly_profit",
        "expected_net_profit",
    )
    np_3q_back = _first_num(
        quarterly,
        "net_profit_3quarters_back",
        "net profit 3quarters back",
        "net_profit_3q_back",
    )
    return {
        "name": symbol,
        "description": description or "",
        "market_cap": _first_num(metrics, "market_capitalization", "market_cap"),
        "sales": _first_num(metrics, "sales", "sales_ttm", "revenue"),
        "peg": _first_num(metrics, "peg_ratio", "peg"),
        "sales_growth_5y": _first_num(
            metrics, "sales_growth_5years", "sales_growth_5y"
        ),
        "operating_profit_growth": _first_num(
            metrics, "operating_profit_growth", "opm_growth"
        ),
        "eps_growth_5y": _first_num(metrics, "eps_growth_5years", "eps_growth_5y"),
        "roe_5y": _first_num(
            metrics, "average_return_on_equity_5years", "average_roe_5y"
        ),
        "roce_or_roic": _first_num(
            metrics,
            "average_return_on_capital_employed_3years",
            "average_roce_3y",
            "roce_percent",
        ),
        "expected_quarterly_profit": expected_q_np,
        "profit_3q_back": np_3q_back,
        "quarterly_profit_growth": _pct_change(expected_q_np, np_3q_back),
    }


def screen_india_garp(
    universe: pd.DataFrame,
    *,
    limit: int,
    workers: int,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    items = [
        (str(row["name"]), str(row.get("description") or ""))
        for _, row in universe.iterrows()
        if row.get("name")
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                cached_json_call,
                "garp_india",
                ("india", symbol),
                ttl_seconds=cache_ttl,
                refresh=refresh,
                fetch=lambda symbol=symbol: _fetch_india_sections(symbol),
            ): (symbol, description)
            for symbol, description in items
        }
        for future in as_completed(futures):
            symbol, description = futures[future]
            try:
                row = _india_row(symbol, description, future.result())
            except Exception:
                continue
            if _passes_garp(row, INDIA_THRESHOLDS):
                rows.append(row)
    return add_garp_score(pd.DataFrame(rows)).head(limit)


def _us_row(symbol: str, description: str | None) -> dict[str, Any]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    income = ticker.income_stmt
    estimates = ticker.earnings_estimate

    revenue = _series_from_statement(income, ["Total Revenue"])
    operating = _series_from_statement(
        income, ["Operating Income", "Operating Income As Reported"]
    )
    net_income = _series_from_statement(
        income, ["Net Income", "Net Income Common Stockholders"]
    )
    equity = _series_from_statement(income, ["Stockholders Equity", "Total Equity"])
    ebit = _series_from_statement(income, ["EBIT", "Operating Income"])
    tax_rate = _series_from_statement(income, ["Tax Rate For Calcs"])
    debt = pd.Series(dtype=float)
    try:
        balance = ticker.balance_sheet
        debt = _series_from_statement(balance, ["Total Debt"])
        equity = _series_from_statement(
            balance, ["Stockholders Equity", "Total Stockholder Equity"]
        )
    except Exception:
        balance = pd.DataFrame()

    quarterly_eps_growth = None
    expected_eps = None
    year_ago_eps = None
    if estimates is not None and not estimates.empty and "0q" in estimates.index:
        expected_eps = _num(estimates.loc["0q"].get("avg"))
        year_ago_eps = _num(estimates.loc["0q"].get("yearAgoEps"))
        quarterly_eps_growth = _pct_change(expected_eps, year_ago_eps)

    latest_revenue = _num(revenue.iloc[0]) if not revenue.empty else None
    oldest_revenue = (
        _num(revenue.iloc[min(len(revenue) - 1, 4)]) if len(revenue) else None
    )
    latest_op = _num(operating.iloc[0]) if not operating.empty else None
    old_op = (
        _num(operating.iloc[min(len(operating) - 1, 1)]) if len(operating) else None
    )
    latest_ni = _num(net_income.iloc[0]) if not net_income.empty else None
    old_ni = (
        _num(net_income.iloc[min(len(net_income) - 1, 4)]) if len(net_income) else None
    )

    tax = _num(tax_rate.iloc[0]) if not tax_rate.empty else 0.21
    nopat = ebit * (1.0 - float(tax or 0.21))
    invested_capital = debt.add(equity, fill_value=0)
    roic = _average_ratio(nopat, invested_capital, 3)

    return {
        "name": symbol,
        "description": description or info.get("shortName") or "",
        "market_cap": _num(info.get("marketCap")),
        "sales": latest_revenue,
        "peg": _num(info.get("trailingPegRatio") or info.get("pegRatio")),
        "sales_growth_5y": _cagr(latest_revenue, oldest_revenue, 4),
        "operating_profit_growth": _pct_change(latest_op, old_op),
        "eps_growth_5y": _cagr(latest_ni, old_ni, 4),
        "roe_5y": _average_ratio(net_income, equity, 5),
        "roce_or_roic": roic,
        "expected_quarterly_profit": expected_eps,
        "profit_3q_back": year_ago_eps,
        "quarterly_profit_growth": quarterly_eps_growth,
    }


def screen_us_garp(
    universe: pd.DataFrame,
    *,
    limit: int,
    workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    items = [
        (str(row["name"]), str(row.get("description") or ""))
        for _, row in universe.iterrows()
        if row.get("name")
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_us_row, symbol, description): (symbol, description)
            for symbol, description in items
        }
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception:
                continue
            if _passes_garp(row, US_THRESHOLDS):
                rows.append(row)
    return add_garp_score(pd.DataFrame(rows)).head(limit)
