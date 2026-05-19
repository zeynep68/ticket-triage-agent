VERSION_MAPPING = {
    "dataset-tickets-multi-lang3-4k.csv": "v3",
    "dataset-tickets-multi-lang-4-20k.csv": "v4",
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv": "v5",
    "dataset-tickets-german_normalized.csv": "de_norm",
    "dataset-tickets-german_normalized_50_5_2.csv": "de_norm_50_5_2",
}

GERMAN_ONLY_FILES = {
    "dataset-tickets-german_normalized.csv",
    "dataset-tickets-german_normalized_50_5_2.csv",
}

# Semantic column renames.
COLUMN_RENAMES = {
    "text": "body",  # v3 uses 'text' instead of 'body'
}

TARGET_LANGUAGES = {"en", "de"}

# When deduplicating identical (subject, body) across versions, keep the row
# from the highest-priority version. v5 is the newest/largest curated set.
VERSION_PRIORITY = {
    "v5": 5,
    "v4": 4,
    "v3": 3,
    "de_norm_50_5_2": 2,
    "de_norm": 1,
}
