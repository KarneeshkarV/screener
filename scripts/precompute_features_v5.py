"""Pre-compute features for v5 expanded dataset."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from screener.ml_signal_v5 import V5FeatureExtractor


def main():
    data_dir = Path(__file__).parent / "training_data_v5"
    cache_path = data_dir / "v5_features.pkl"

    print("Loading trades and bars...")
    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    trades = trades_data["trades"]
    bars_data = bars_json.get("bars", bars_json)

    bars_by_tv = {}
    for key, records in bars_data.items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        bars_by_tv[key] = df

    # Build simple ticker lookup (market:sym -> sym)
    bars_lookup = {}
    for key, df in bars_by_tv.items():
        if ":" in key:
            _, sym = key.split(":", 1)
        else:
            sym = key
        bars_lookup[sym] = df

    print(f"Loaded {len(bars_lookup)} symbols, {len(trades)} trades")

    # Pre-compute features
    print("Pre-computing features...")
    extractor = V5FeatureExtractor()
    features_cache = {}

    for sym, bars in bars_lookup.items():
        if bars is None or bars.empty:
            continue
        features_cache[sym] = extractor.extract(bars)

    with open(cache_path, "wb") as f:
        pickle.dump({"features": features_cache}, f)

    print(f"Features cached to {cache_path}")


if __name__ == "__main__":
    main()
