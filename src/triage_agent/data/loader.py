import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from triage_agent.data.constants import (
    COLUMN_RENAMES,
    GERMAN_ONLY_FILES,
    VERSION_MAPPING,
    VERSION_PRIORITY,
)
from triage_agent.schemas import CanonicalTicket

log = logging.getLogger(__name__)


@dataclass
class FileTickets:
    """Result of parsing one CSV: validated tickets and rows that failed validation."""

    tickets: list[CanonicalTicket] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)


def _hash_id(filename: str, row_index: int) -> int:
    h = hashlib.md5(f"{filename}:{row_index}".encode())
    return int(h.hexdigest()[:12], 16)


def load_single_csv(path: Path) -> FileTickets:
    """Load one CSV, normalize columns, validate each row via Pydantic.

    Invalid rows are collected in `rejected_rows` instead of being dropped
    silently, so data-quality issues remain visible downstream.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns=COLUMN_RENAMES)

    if path.name in GERMAN_ONLY_FILES:
        df["language"] = "de"

    df["version"] = VERSION_MAPPING[path.name]

    result = FileTickets()
    for idx, row in df.iterrows():
        try:
            ticket = CanonicalTicket(
                ticket_id=_hash_id(path.name, int(idx)),
                subject=row.get("subject", "") or "",
                body=row.get("body", "") or "",
                language=row.get("language", "unknown") or "unknown",
                version=row["version"],
                queue=row.get("queue") or None,
                priority=row.get("priority") or None,
                type=row.get("type") or None,
            )
            result.tickets.append(ticket)
        except (ValueError, ValidationError) as e:
            result.rejected_rows.append(
                {
                    "file": path.name,
                    "row_index": int(idx),
                    "error": str(e),
                }
            )
    return result


def dedup_by_version(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (subject, body) pairs, keeping the highest-priority version."""
    df = df.copy()
    df["_rank"] = df["version"].map(VERSION_PRIORITY).fillna(0)
    df = (
        df.sort_values("_rank", ascending=False)
        .drop_duplicates(subset=["subject", "body"], keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return df


def load_all_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every known CSV in `data_dir`, merge, dedup.

    Returns (valid_df, rejected_df). Unknown CSV filenames are skipped with a warning.
    """
    all_valid: list[CanonicalTicket] = []
    all_rejected: list[dict] = []

    expected = set(VERSION_MAPPING.keys())
    found = {p.name for p in data_dir.glob("*.csv")}

    missing = expected - found
    unknown = found - expected
    if missing:
        log.warning(
            "Expected CSV files are missing from the data directory: %s.", missing
        )
    if unknown:
        log.warning(
            "Skipping CSV files not in the version mapping (unknown source): %s.",
            unknown,
        )

    for path in sorted(data_dir.glob("*.csv")):
        if path.name not in VERSION_MAPPING:
            continue
        result = load_single_csv(path)
        all_valid.extend(result.tickets)
        all_rejected.extend(result.rejected_rows)
        log.info(
            "File %s yielded %d valid tickets and %d invalid rows (rows that failed schema validation).",
            path.name,
            len(result.tickets),
            len(result.rejected_rows),
        )

    log.info(
        "Schema validation across all CSVs: %d tickets passed, %d rows failed validation.",
        len(all_valid),
        len(all_rejected),
    )

    valid_df = pd.DataFrame([t.model_dump() for t in all_valid])
    rejected_df = pd.DataFrame(all_rejected)

    if not valid_df.empty:
        pre_dedup_count = len(valid_df)
        valid_df = dedup_by_version(valid_df)
        removed = pre_dedup_count - len(valid_df)
        log.info(
            "Deduplicated to %d unique tickets after removing %d duplicates that appeared across multiple dataset versions.",
            len(valid_df),
            removed,
        )

    return valid_df, rejected_df
