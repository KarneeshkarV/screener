import pandas as pd


def enrich_fundamentals(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market != "india":
        return df

    try:
        from openscreener import Stock
    except ImportError:
        return df

    symbols = df["name"].tolist()
    if not symbols:
        return df

    try:
        batch = Stock.batch(symbols)
        ratios_data = batch.fetch("ratios")
    except Exception:
        return df

    rows = []
    for symbol in symbols:
        data = ratios_data.get(symbol, {})
        rows.append(
            {
                "name": symbol,
                "P/E": data.get("stock_p_e"),
                "ROCE%": data.get("roce_percent"),
                "ROE%": data.get("return_on_equity"),
            }
        )

    fundamentals = pd.DataFrame(rows)
    return df.merge(fundamentals, on="name", how="left")
