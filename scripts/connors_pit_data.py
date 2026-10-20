"""Validate timestamped Connors input bundles without certifying vendor evidence.

All security keys are permanent IDs, never inferred ticker aliases.
The daily universe and calendar must be supplied independently of price coverage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.connors_crash_research import (
    ConnorsMarketData,
    connors_contiguous_indicators,
)


def read_connors_evidence(root: Path, name: str, required: set[str]) -> pd.DataFrame:
    """Read one evidence table and fail on missing columns or empty identities."""
    frame = pd.read_csv(root / name, dtype={"security_id": str})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Connors PIT columns missing in {name}: {sorted(missing)}")
    if "security_id" in frame and (
        frame.security_id.isna().any() or frame.security_id.str.strip().eq("").any()
    ):
        raise ValueError(f"Connors PIT empty security ID in {name}")
    return frame


def connors_evidence_times(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Require explicit timezone offsets, then normalize evidence timestamps to UTC."""
    for column in columns:
        text = frame[column].astype(str)
        if not text.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise ValueError(f"Connors PIT timestamp needs timezone: {column}")
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")


def connors_evidence_sessions(frame: pd.DataFrame) -> None:
    """Require date-only session keys, not intraday or timezone-dependent dates."""
    if not frame.session.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Connors PIT session must be YYYY-MM-DD")
    frame["session"] = pd.to_datetime(frame.session, errors="raise")


def load_connors_pit_bundle(
    root: Path, market: str, end: pd.Timestamp, earliest_start: pd.Timestamp
) -> ConnorsMarketData:
    """Load complete daily evidence; refuse unknown actions and unsupported short claims.

    This validates a supplied data contract, not the truth or completeness of its source.
    Daily fills remain hypothetical even when dated borrow evidence is supplied.
    """
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("market") != market:
        raise ValueError("Connors PIT metadata version or market mismatch")
    if metadata.get("universe_kind") not in {"all_listed", "index"}:
        raise ValueError("Connors PIT universe_kind must be all_listed or index")
    for claim in (
        "calendar_source",
        "universe_source",
        "security_source",
        "price_source",
        "action_source",
    ):
        if not isinstance(metadata.get(claim), str) or not metadata[claim].strip():
            raise ValueError(f"Connors PIT source reference required: {claim}")
    if (
        metadata.get("actions_complete_through") is None
        or pd.Timestamp(metadata["actions_complete_through"]) < end
    ):
        raise ValueError(
            "Connors PIT corporate-action coverage ends before requested end"
        )
    sessions = read_connors_evidence(
        root,
        "sessions.csv",
        {"session", "decision_at", "open_at", "close_at", "available_at"},
    )
    connors_evidence_sessions(sessions)
    connors_evidence_times(
        sessions, ("decision_at", "open_at", "close_at", "available_at")
    )
    if (
        sessions.session.duplicated().any()
        or not sessions.session.is_monotonic_increasing
    ):
        raise ValueError("Connors PIT calendar must be unique and sorted")
    if not (
        (sessions.available_at <= sessions.decision_at)
        & (sessions.decision_at <= sessions.open_at)
        & (sessions.open_at < sessions.close_at)
    ).all():
        raise ValueError("Connors PIT invalid calendar availability or session times")
    if (
        len(sessions) == 0
        or sessions.session.max() < end
        or sessions.session.min() >= earliest_start
    ):
        raise ValueError("Connors PIT calendar lacks requested coverage or warmup")
    exchange_timezone = "America/New_York" if market == "us" else "Asia/Kolkata"
    for column in ("decision_at", "open_at", "close_at"):
        local_dates = sessions[column].dt.tz_convert(exchange_timezone).dt.date
        if not local_dates.eq(sessions.session.dt.date).all():
            raise ValueError(
                "Connors PIT session label disagrees with exchange-local timestamp"
            )
    if (
        not sessions.decision_at.is_monotonic_increasing
        or not sessions.close_at.is_monotonic_increasing
    ):
        raise ValueError("Connors PIT calendar timestamps must increase")
    if not (
        sessions.close_at.iloc[:-1].to_numpy()
        < sessions.decision_at.iloc[1:].to_numpy()
    ).all():
        raise ValueError("Connors PIT sessions overlap next decision")
    # Keep all supplied warmup; never infer sessions from a benchmark's prices.
    sessions = sessions[sessions.session <= end].set_index("session")
    calendar = pd.DatetimeIndex(sessions.index)
    securities = read_connors_evidence(
        root,
        "securities.csv",
        {"security_id", "available_at", "listed_on", "delisted_on"},
    )
    securities["listed_on"] = pd.to_datetime(securities.listed_on, errors="raise")
    securities["delisted_on"] = pd.to_datetime(securities.delisted_on, errors="raise")
    if (
        securities.listed_on.isna().any()
        or (securities.delisted_on < securities.listed_on).any()
    ):
        raise ValueError("Connors PIT invalid listing interval")
    connors_evidence_times(securities, ("available_at",))
    if securities.security_id.duplicated().any():
        raise ValueError("Connors PIT duplicate permanent security ID")
    symbols = sorted(securities.security_id)
    security_times = securities.set_index("security_id").available_at
    shape = (len(calendar), len(symbols))
    symbol_indices = {s: j for j, s in enumerate(symbols)}
    keys = ("open", "high", "low", "close", "turnover", "crsi", "volatility")
    arrays = {key: np.full(shape, np.nan) for key in keys}
    arrays.update(
        {
            "dividend": np.zeros(shape),
            "split": np.ones(shape),
            "complete_window": np.zeros(shape, dtype=bool),
            "entry_quality": np.zeros(shape, dtype=bool),
            "prior_bar_available": np.zeros(shape, dtype=bool),
            "prior_quote_available": np.zeros(shape, dtype=bool),
        }
    )

    def evidence_table(name: str, columns: set[str]) -> pd.DataFrame:
        frame = read_connors_evidence(root, name, columns | {"session", "available_at"})
        connors_evidence_sessions(frame)
        connors_evidence_times(frame, ("available_at",))
        if frame.duplicated(["session", "security_id"]).any():
            raise ValueError(f"Connors PIT duplicate evidence in {name}")
        if (
            not frame.security_id.isin(symbols).all()
            or not frame.session.isin(calendar).all()
        ):
            raise ValueError(f"Connors PIT unknown security or session in {name}")
        return frame

    # Exactly one manifest row per supplied session, including empty-universe days.
    universe_days = read_connors_evidence(
        root, "universe_sessions.csv", {"session", "available_at", "member_count"}
    )
    connors_evidence_sessions(universe_days)
    connors_evidence_times(universe_days, ("available_at",))
    if universe_days.session.duplicated().any() or set(universe_days.session) != set(
        calendar
    ):
        raise ValueError(
            "Connors PIT universe coverage must match every calendar session"
        )
    universe = evidence_table("universe.csv", {"security_id", "ticker"})
    if (
        universe.ticker.isna().any()
        or universe.ticker.astype(str).str.contains("DUMMY", case=False).any()
    ):
        raise ValueError("Connors PIT invalid ticker or corporate-action placeholder")
    if universe.duplicated(["session", "ticker"]).any():
        raise ValueError(
            "Connors PIT ticker maps to multiple security IDs on one session"
        )
    membership = np.zeros(shape, dtype=bool)
    for row in universe_days.itertuples():
        if row.available_at > sessions.loc[row.session, "decision_at"]:
            raise ValueError("Connors PIT universe manifest unavailable at decision")
        actual = universe[universe.session == row.session]
        listed = securities[
            (securities.listed_on <= row.session)
            & (securities.delisted_on.isna() | (securities.delisted_on >= row.session))
        ]
        if not actual.security_id.isin(listed.security_id).all():
            raise ValueError("Connors PIT universe member outside listing interval")
        if metadata["universe_kind"] == "all_listed" and set(actual.security_id) != set(
            listed.security_id
        ):
            raise ValueError("Connors PIT all-listed universe omits a listed security")
        if (
            not np.isfinite(row.member_count)
            or row.member_count < 0
            or row.member_count != len(actual)
        ):
            raise ValueError("Connors PIT universe member count mismatch")
    for row in universe.itertuples():
        decision = sessions.loc[row.session, "decision_at"]
        if row.available_at > decision or security_times[row.security_id] > decision:
            raise ValueError("Connors PIT future universe or security identity")
        membership[calendar.get_loc(row.session), symbol_indices[row.security_id]] = (
            True
        )

    bars = evidence_table(
        "bars.csv", {"security_id", "open", "high", "low", "close", "turnover"}
    )
    listing_bounds = securities.set_index("security_id")
    for row in bars.itertuples():
        bounds = listing_bounds.loc[row.security_id]
        if row.session < bounds.listed_on or (
            pd.notna(bounds.delisted_on) and row.session > bounds.delisted_on
        ):
            raise ValueError("Connors PIT quote outside listing interval")
    # Delisting settlement is unsupported. Refuse it even without an action row.
    for row in securities.itertuples():
        if pd.notna(row.delisted_on) and row.delisted_on <= end:
            j = symbol_indices[row.security_id]
            arrays.setdefault("delisted", np.zeros(shape, dtype=bool))[:, j] = (
                calendar > row.delisted_on
            )
    actions = evidence_table(
        "actions.csv", {"security_id", "split", "dividend", "unsupported"}
    )
    for row in actions.itertuples():
        if row.available_at > sessions.loc[row.session, "decision_at"]:
            raise ValueError("Connors PIT corporate action unknown at decision")
        if row.unsupported != 0:
            raise ValueError(
                "Connors PIT unsupported corporate action requires verified settlement"
            )
        if (
            not np.isfinite(row.split)
            or row.split <= 0
            or not np.isfinite(row.dividend)
            or row.dividend < 0
        ):
            raise ValueError("Connors PIT invalid split or dividend")
        i, j = calendar.get_loc(row.session), symbol_indices[row.security_id]
        arrays["split"][i, j], arrays["dividend"][i, j] = row.split, row.dividend
    bar_times: dict[str, pd.Series] = {}
    invalid_counts: dict[str, int] = {}
    anomalies = []
    for symbol, group in bars.groupby("security_id"):
        j = symbol_indices[symbol]
        group = group.set_index("session").reindex(calendar)
        numeric = group[list(keys[:5])].astype(float)
        valid = np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)
        valid &= (numeric.high >= numeric[["open", "close"]].max(axis=1)) & (
            numeric.low <= numeric[["open", "close"]].min(axis=1)
        )
        # Outcome bars must not claim to have been available before that bar closed.
        if (
            group.available_at.dropna()
            < sessions.loc[group.available_at.dropna().index, "close_at"]
        ).any():
            raise ValueError("Connors PIT full daily bar available before close")
        bar_times[symbol] = group.available_at.where(valid)
        for key in keys[:5]:
            arrays[key][:, j] = numeric[key].where(valid).to_numpy()
        invalid_counts[symbol] = int((~valid & group.available_at.notna()).sum())
        # Adjust only when an action occurs. Future action rows never rescale past signals.
        close = pd.Series(arrays["close"][:, j], index=calendar)
        split = pd.Series(arrays["split"][:, j], index=calendar)
        dividend = pd.Series(arrays["dividend"][:, j], index=calendar)
        gross_return = (close + dividend) * split / close.shift(1)
        signal = pd.Series(np.nan, index=calendar)
        for _, segment in close[close.notna()].groupby(
            close.isna().cumsum()[close.notna()]
        ):
            signal.loc[segment.index] = (
                gross_return.loc[segment.index].fillna(1).cumprod()
            )
        for stamp, value in gross_return.sub(1).items():
            if np.isfinite(value) and abs(value) > 0.5:
                anomalies.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "date": str(stamp.date()),
                        "adjusted_close_return": value,
                    }
                )
        features = connors_contiguous_indicators(signal)
        for key in ("crsi", "volatility"):
            arrays[key][:, j] = features[key].to_numpy()
        arrays["complete_window"][:, j] = close.notna().rolling(101).sum().eq(101)
        arrays["entry_quality"][:, j] = arrays["complete_window"][:, j]
        # Wilder smoothing retains every observation since the latest missing bar.
        latest_arrival = None
        for i in range(1, len(calendar)):
            prior_time = bar_times[symbol].iloc[i - 1]
            if pd.isna(prior_time):
                latest_arrival = None
                continue
            latest_arrival = (
                prior_time
                if latest_arrival is None
                else max(latest_arrival, prior_time)
            )
            arrays["prior_quote_available"][i, j] = (
                prior_time <= sessions.decision_at.iloc[i]
            )
            arrays["prior_bar_available"][i, j] = (
                latest_arrival <= sessions.decision_at.iloc[i]
            )

    # Borrow evidence is optional; its absence must not become assumed availability.
    permissions_path = root / "short_permissions.csv"
    if permissions_path.exists():
        if not metadata.get("borrow_source"):
            raise ValueError("Connors PIT borrow source reference required")
        permissions = evidence_table(
            "short_permissions.csv",
            {
                "security_id",
                "max_quantity",
                "annual_rate",
                "contract_id",
                "valid_until",
            },
        )
        connors_evidence_times(permissions, ("valid_until",))
        arrays["short_capacity"] = np.zeros(shape)
        arrays["borrow_rate"] = np.zeros(shape)
        for row in permissions.itertuples():
            decision = sessions.loc[row.session, "decision_at"]
            if (
                row.available_at > decision
                or row.valid_until < sessions.loc[row.session, "close_at"]
            ):
                raise ValueError("Connors PIT late or expired short permission")
            if (
                not np.isfinite(row.max_quantity)
                or row.max_quantity < 0
                or not np.isfinite(row.annual_rate)
                or row.annual_rate < 0
            ):
                raise ValueError("Connors PIT invalid short capacity or borrow rate")
            if market == "india" and (
                pd.isna(row.contract_id) or not str(row.contract_id).strip()
            ):
                raise ValueError(
                    "Connors PIT India short permission requires SLB contract ID"
                )
            i, j = calendar.get_loc(row.session), symbol_indices[row.security_id]
            arrays["short_capacity"][i, j] = row.max_quantity
            arrays["borrow_rate"][i, j] = row.annual_rate

    coverage, missing = [], []
    for j, symbol in enumerate(symbols):
        expected = membership[:, j] & (calendar >= earliest_start)
        present = np.isfinite(arrays["close"][:, j])
        coverage.append(
            {
                "market": market,
                "symbol": symbol,
                "status": "timestamped_nominal",
                "invalid_bars": invalid_counts.get(symbol, 0),
                "expected_member_sessions": int(expected.sum()),
                "missing_member_sessions": int((expected & ~present).sum()),
                "unready_indicator_sessions": int(
                    (expected & present & ~arrays["complete_window"][:, j]).sum()
                ),
            }
        )
        for stamp in calendar[expected & ~present]:
            missing.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "date": str(stamp.date()),
                    "location": "missing_or_invalid_quote",
                }
            )
    sources = [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.iterdir())
        if p.is_file()
    ]
    data = ConnorsMarketData(
        market,
        calendar,
        symbols,
        arrays,
        membership,
        pd.DataFrame(coverage),
        pd.DataFrame(missing, columns=["market", "symbol", "date", "location"]),
        universe_days,
        pd.DataFrame(
            anomalies, columns=["market", "symbol", "date", "adjusted_close_return"]
        ),
        sources,
    )
    data.evidence = {
        "classification": "timestamped_signal_research",
        "universe_kind": metadata["universe_kind"],
        "source_assertions_verified": False,
        "borrow_evidence_supplied": permissions_path.exists(),
        "executable": False,
        "delisting_dates": {
            row.security_id: str(row.delisted_on.date())
            for row in securities.itertuples()
            if pd.notna(row.delisted_on)
        },
        "remaining_limits": [
            "Source completeness and provenance need independent verification",
            "Daily touch fills do not establish executable fills",
            "Restrictions, margin, settlement, recalls and partial fills are not fully modeled",
        ],
    }
    return data
