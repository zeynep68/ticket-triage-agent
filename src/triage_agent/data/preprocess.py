import re

import pandas as pd

from triage_agent.data.constants import TARGET_LANGUAGES


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def merge_subject_body(row: pd.Series) -> str:
    subject = (row.get("subject") or "").strip()
    body = (row.get("body") or "").strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def cap_length(text: str, max_chars: int = 2000) -> str:
    """Keep head + tail when text is too long — final question often sits at the end."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n[...]\n" + text[-200:]


def filter_languages(df: pd.DataFrame, langs: set[str] = TARGET_LANGUAGES) -> pd.DataFrame:
    return df[df["language"].isin(langs)].reset_index(drop=True)


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df.apply(merge_subject_body, axis=1)
    df["text"] = df["text"].apply(normalize_whitespace)
    df["text"] = df["text"].apply(cap_length)
    df["text_length"] = df["text"].str.len()
    df = df[df["text_length"] > 0].reset_index(drop=True)
    return df
