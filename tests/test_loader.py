import tempfile
from pathlib import Path

import pandas as pd

from triage_agent.data.loader import dedup_by_version, load_single_csv


def _write_csv(rows: list[dict], filename: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    path = tmp / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_v3_text_column_aliased_to_body():
    path = _write_csv(
        [{"Subject": "Help", "text": "Account locked", "language": "en"}],
        "dataset-tickets-multi-lang3-4k.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 1
    assert result.tickets[0].body == "Account locked"
    assert result.tickets[0].subject == "Help"
    assert result.tickets[0].version == "v3"


def test_uppercase_columns_are_normalized():
    path = _write_csv(
        [{"SUBJECT": "Help", "BODY": "Account locked", "LANGUAGE": "en"}],
        "dataset-tickets-multi-lang-4-20k.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 1
    assert result.tickets[0].subject == "Help"
    assert result.tickets[0].body == "Account locked"


def test_german_only_file_gets_language_hardcoded():
    path = _write_csv(
        [{"subject": "Hilfe", "body": "Mein Konto ist gesperrt"}],
        "dataset-tickets-german_normalized.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 1
    assert result.tickets[0].language == "de"
    assert result.tickets[0].version == "de_norm"


def test_both_subject_and_body_empty_is_rejected():
    path = _write_csv(
        [{"subject": "", "body": "", "language": "en"}],
        "dataset-tickets-multi-lang-4-20k.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 0
    assert len(result.rejected_rows) == 1
    assert "cannot both be empty" in result.rejected_rows[0]["error"].lower()


def test_empty_body_with_subject_set_is_kept():
    path = _write_csv(
        [{"subject": "Help me", "body": "", "language": "en"}],
        "dataset-tickets-multi-lang-4-20k.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 1
    assert result.tickets[0].subject == "Help me"
    assert result.tickets[0].body == ""


def test_empty_subject_with_body_set_is_kept():
    path = _write_csv(
        [{"subject": "", "body": "Mein Auto wurde gestohlen", "language": "de"}],
        "dataset-tickets-german_normalized.csv",
    )
    result = load_single_csv(path)
    assert len(result.tickets) == 1
    assert result.tickets[0].subject == ""
    assert result.tickets[0].body == "Mein Auto wurde gestohlen"


def test_language_is_normalized_to_two_letter_lowercase():
    path = _write_csv(
        [{"subject": "Hi", "body": "Hello", "language": "EN-US"}],
        "dataset-tickets-multi-lang-4-20k.csv",
    )
    result = load_single_csv(path)
    assert result.tickets[0].language == "en"


def test_dedup_keeps_higher_version():
    df = pd.DataFrame(
        [
            {"subject": "X", "body": "Y", "version": "v3", "language": "en"},
            {"subject": "X", "body": "Y", "version": "v5", "language": "en"},
        ]
    )
    deduped = dedup_by_version(df)
    assert len(deduped) == 1
    assert deduped.iloc[0]["version"] == "v5"


def test_dedup_preserves_different_bodies():
    df = pd.DataFrame(
        [
            {"subject": "X", "body": "Y", "version": "v3", "language": "en"},
            {"subject": "X", "body": "Z", "version": "v5", "language": "en"},
        ]
    )
    deduped = dedup_by_version(df)
    assert len(deduped) == 2
