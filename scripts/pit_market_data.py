"""Validate timestamped market input bundles without certifying vendor evidence.

All security keys are permanent IDs, never inferred ticker aliases.
The daily universe and calendar must be supplied independently of price coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PitMarketData:
    """Daily nominal prices and evidence on an explicit session calendar."""

    market: str
    calendar: pd.DatetimeIndex
    symbols: list[str]
    arrays: dict[str, np.ndarray]
    membership: np.ndarray
    coverage: pd.DataFrame
    missing_sessions: pd.DataFrame
    observations: pd.DataFrame
    anomalies: pd.DataFrame
    sources: list[dict]
    evidence: dict = field(
        default_factory=lambda: {
            "classification": "observed_index_proxy",
            "executable": False,
        }
    )


def read_pit_evidence(root: Path, name: str, required: set[str]) -> pd.DataFrame:
    """Read one evidence table and fail on missing columns or empty identities."""
    frame = pd.read_csv(root / name, dtype={"security_id": str})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"PIT market data columns missing in {name}: {sorted(missing)}"
        )
    if "security_id" in frame and (
        frame.security_id.isna().any() or frame.security_id.str.strip().eq("").any()
    ):
        raise ValueError(f"PIT market data empty security ID in {name}")
    return frame


def pit_evidence_times(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Require explicit timezone offsets, then normalize evidence timestamps to UTC."""
    for column in columns:
        text = frame[column].astype(str)
        if not text.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise ValueError(f"PIT market data timestamp needs timezone: {column}")
        frame[column] = pd.to_datetime(
            frame[column], utc=True, errors="raise", format="ISO8601"
        )


def pit_evidence_sessions(frame: pd.DataFrame) -> None:
    """Require date-only session keys, not intraday or timezone-dependent dates."""
    if not frame.session.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("PIT market data session must be YYYY-MM-DD")
    frame["session"] = pd.to_datetime(frame.session, errors="raise")


def load_pit_market_bundle(
    root: Path,
    market: str,
    end: pd.Timestamp,
    earliest_start: pd.Timestamp,
    *,
    history_sessions: int = 1,
) -> PitMarketData:
    """Load complete daily evidence; refuse unknown actions and unsupported short claims.

    This validates a supplied data contract, not the truth or completeness of its source.
    No indicator, order, or fill model is applied here.
    history_sessions specifies the caller's required contiguous price history.
    """
    if history_sessions < 1:
        raise ValueError("PIT market data history_sessions must be positive")
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("market") != market:
        raise ValueError("PIT market data metadata version or market mismatch")
    if metadata.get("universe_kind") not in {"all_listed", "index"}:
        raise ValueError("PIT market data universe_kind must be all_listed or index")
    for claim in (
        "calendar_source",
        "universe_source",
        "security_source",
        "price_source",
        "action_source",
    ):
        if not isinstance(metadata.get(claim), str) or not metadata[claim].strip():
            raise ValueError(f"PIT market data source reference required: {claim}")
    if (
        metadata.get("actions_complete_through") is None
        or pd.Timestamp(metadata["actions_complete_through"]) < end
    ):
        raise ValueError(
            "PIT market data corporate-action coverage ends before requested end"
        )
    sessions = read_pit_evidence(
        root,
        "sessions.csv",
        {"session", "decision_at", "open_at", "close_at", "available_at"},
    )
    pit_evidence_sessions(sessions)
    pit_evidence_times(sessions, ("decision_at", "open_at", "close_at", "available_at"))
    if (
        sessions.session.duplicated().any()
        or not sessions.session.is_monotonic_increasing
    ):
        raise ValueError("PIT market data calendar must be unique and sorted")
    if not (
        (sessions.available_at <= sessions.decision_at)
        & (sessions.decision_at <= sessions.open_at)
        & (sessions.open_at < sessions.close_at)
    ).all():
        raise ValueError(
            "PIT market data invalid calendar availability or session times"
        )
    if (
        len(sessions) == 0
        or sessions.session.max() < end
        or sessions.session.min() >= earliest_start
    ):
        raise ValueError("PIT market data calendar lacks requested coverage or warmup")
    warmup_sessions = int((sessions.session < earliest_start).sum())
    if warmup_sessions < history_sessions:
        raise ValueError(
            f"PIT market data calendar requires {history_sessions} pre-start sessions; found {warmup_sessions}"
        )
    exchange_timezone = "America/New_York" if market == "us" else "Asia/Kolkata"
    for column in ("decision_at", "open_at", "close_at"):
        local_dates = sessions[column].dt.tz_convert(exchange_timezone).dt.date
        if not local_dates.eq(sessions.session.dt.date).all():
            raise ValueError(
                "PIT market data session label disagrees with exchange-local timestamp"
            )
    if (
        not sessions.decision_at.is_monotonic_increasing
        or not sessions.close_at.is_monotonic_increasing
    ):
        raise ValueError("PIT market data calendar timestamps must increase")
    if not (
        sessions.close_at.iloc[:-1].to_numpy()
        < sessions.decision_at.iloc[1:].to_numpy()
    ).all():
        raise ValueError("PIT market data sessions overlap next decision")
    # Keep all supplied warmup; never infer sessions from a benchmark's prices.
    sessions = sessions[sessions.session <= end].set_index("session")
    calendar = pd.DatetimeIndex(sessions.index)
    securities = read_pit_evidence(
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
        raise ValueError("PIT market data invalid listing interval")
    pit_evidence_times(securities, ("available_at",))
    if securities.security_id.duplicated().any():
        raise ValueError("PIT market data duplicate permanent security ID")
    symbols = sorted(securities.security_id)
    security_master = securities.set_index("security_id")
    shape = (len(calendar), len(symbols))
    symbol_indices = {s: j for j, s in enumerate(symbols)}
    keys = ("open", "high", "low", "close", "turnover", "signal_close")
    arrays = {key: np.full(shape, np.nan) for key in keys}
    arrays.update(
        {
            "dividend": np.zeros(shape),
            "split": np.ones(shape),
            "complete_window": np.zeros(shape, dtype=bool),
            "prior_bar_available": np.zeros(shape, dtype=bool),
            "prior_quote_available": np.zeros(shape, dtype=bool),
        }
    )

    def evidence_table(name: str, columns: set[str]) -> pd.DataFrame:
        frame = read_pit_evidence(root, name, columns | {"session", "available_at"})
        pit_evidence_sessions(frame)
        pit_evidence_times(frame, ("available_at",))
        if frame.duplicated(["session", "security_id"]).any():
            raise ValueError(f"PIT market data duplicate evidence in {name}")
        if (
            not frame.security_id.isin(symbols).all()
            or not frame.session.isin(calendar).all()
        ):
            raise ValueError(f"PIT market data unknown security or session in {name}")
        return frame

    # Exactly one manifest row per supplied session, including empty-universe days.
    universe_days = read_pit_evidence(
        root, "universe_sessions.csv", {"session", "available_at", "member_count"}
    )
    pit_evidence_sessions(universe_days)
    pit_evidence_times(universe_days, ("available_at",))
    if universe_days.session.duplicated().any() or set(universe_days.session) != set(
        calendar
    ):
        raise ValueError(
            "PIT market data universe coverage must match every calendar session"
        )
    universe = evidence_table("universe.csv", {"security_id", "ticker"})
    if (
        universe.ticker.isna().any()
        or universe.ticker.astype(str).str.contains("DUMMY", case=False).any()
    ):
        raise ValueError(
            "PIT market data invalid ticker or corporate-action placeholder"
        )
    if universe.duplicated(["session", "ticker"]).any():
        raise ValueError(
            "PIT market data ticker maps to multiple security IDs on one session"
        )
    membership = np.zeros(shape, dtype=bool)
    if (
        universe_days.available_at > universe_days.session.map(sessions.decision_at)
    ).any():
        raise ValueError("PIT market data universe manifest unavailable at decision")
    listed_on = universe.security_id.map(security_master.listed_on)
    delisted_on = universe.security_id.map(security_master.delisted_on)
    if ((universe.session < listed_on) | (universe.session > delisted_on)).any():
        raise ValueError("PIT market data universe member outside listing interval")
    member_counts = universe.groupby("session").size().reindex(calendar, fill_value=0)
    if metadata["universe_kind"] == "all_listed":
        # Unique valid member IDs plus matching counts establish set equality.
        # Count listing intervals once, rather than filtering the master every session.
        listing_changes = np.zeros(len(calendar) + 1, dtype=np.int64)
        first = calendar.searchsorted(securities.listed_on, side="left")
        last = calendar.searchsorted(
            securities.delisted_on.fillna(calendar[-1]), side="right"
        )
        np.add.at(listing_changes, first, 1)
        np.add.at(listing_changes, last, -1)
        if not np.array_equal(member_counts.to_numpy(), listing_changes.cumsum()[:-1]):
            raise ValueError(
                "PIT market data all-listed universe omits a listed security"
            )
    declared_counts = universe_days.set_index("session").member_count.reindex(calendar)
    if not (
        np.isfinite(declared_counts)
        & declared_counts.ge(0)
        & declared_counts.eq(member_counts)
    ).all():
        raise ValueError("PIT market data universe member count mismatch")
    decision_times = universe.session.map(sessions.decision_at)
    if (
        (universe.available_at > decision_times)
        | (universe.security_id.map(security_master.available_at) > decision_times)
    ).any():
        raise ValueError("PIT market data future universe or security identity")
    membership[
        calendar.get_indexer(universe.session),
        universe.security_id.map(symbol_indices).to_numpy(dtype=int),
    ] = True

    bars = evidence_table(
        "bars.csv", {"security_id", "open", "high", "low", "close", "turnover"}
    )
    if (
        (bars.session < bars.security_id.map(security_master.listed_on))
        | (bars.session > bars.security_id.map(security_master.delisted_on))
    ).any():
        raise ValueError("PIT market data quote outside listing interval")
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
            raise ValueError("PIT market data corporate action unknown at decision")
        if row.unsupported != 0:
            raise ValueError(
                "PIT market data unsupported corporate action requires verified settlement"
            )
        if (
            not np.isfinite(row.split)
            or row.split <= 0
            or not np.isfinite(row.dividend)
            or row.dividend < 0
        ):
            raise ValueError("PIT market data invalid split or dividend")
        i, j = calendar.get_loc(row.session), symbol_indices[row.security_id]
        arrays["split"][i, j], arrays["dividend"][i, j] = row.split, row.dividend
    decision_ns = sessions.decision_at.array.as_unit("ns").asi8
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
            raise ValueError("PIT market data full daily bar available before close")
        bar_times = group.available_at.where(valid)
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
        arrays["signal_close"][:, j] = signal.to_numpy()
        arrays["complete_window"][:, j] = (
            close.notna().rolling(history_sessions).sum().eq(history_sessions)
        )
        # Recursive indicators can retain every observation since the latest missing bar.
        # Segmented cumulative maxima check the whole chain in linear time.
        arrived = bar_times.notna().to_numpy()
        arrival_ns = pd.Series(bar_times.array.as_unit("ns").asi8, index=calendar)
        latest_arrival = arrival_ns.groupby((~arrived).cumsum()).cummax().to_numpy()
        arrays["prior_quote_available"][1:, j] = arrived[:-1] & (
            arrival_ns.to_numpy()[:-1] <= decision_ns[1:]
        )
        arrays["prior_bar_available"][1:, j] = arrived[:-1] & (
            latest_arrival[:-1] <= decision_ns[1:]
        )

    # Borrow evidence is optional; its absence must not become assumed availability.
    permissions_path = root / "short_permissions.csv"
    if permissions_path.exists():
        if not metadata.get("borrow_source"):
            raise ValueError("PIT market data borrow source reference required")
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
        pit_evidence_times(permissions, ("valid_until",))
        arrays["short_capacity"] = np.zeros(shape)
        arrays["borrow_rate"] = np.zeros(shape)
        for row in permissions.itertuples():
            decision = sessions.loc[row.session, "decision_at"]
            if (
                row.available_at > decision
                or row.valid_until < sessions.loc[row.session, "close_at"]
            ):
                raise ValueError("PIT market data late or expired short permission")
            if (
                not np.isfinite(row.max_quantity)
                or row.max_quantity < 0
                or not np.isfinite(row.annual_rate)
                or row.annual_rate < 0
            ):
                raise ValueError(
                    "PIT market data invalid short capacity or borrow rate"
                )
            if market == "india" and (
                pd.isna(row.contract_id) or not str(row.contract_id).strip()
            ):
                raise ValueError(
                    "PIT market data India short permission requires SLB contract ID"
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
                "incomplete_history_sessions": int(
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
    data = PitMarketData(
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
