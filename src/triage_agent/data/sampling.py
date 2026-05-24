import logging

import pandas as pd

log = logging.getLogger(__name__)


def stratified_sample(
    df: pd.DataFrame,
    n: int = 2000,
    stratify_by: str = "queue",
    seed: int = 42,
) -> pd.DataFrame:
    """Balance the sample across `stratify_by` values, capped at group size.

    Small strata are not oversampled. If the resulting sample is smaller than `n`,
    the gap is filled with a random draw from the remainder.
    """
    if stratify_by not in df.columns or df[stratify_by].isna().all():
        log.warning(
            "No '%s' column or all-NaN — using random sample instead", stratify_by
        )
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    counts = df[stratify_by].value_counts()
    target = n // max(len(counts), 1)

    samples = []
    for _, group in df.groupby(stratify_by):
        take = min(target, len(group))
        samples.append(group.sample(n=take, random_state=seed))

    sampled = pd.concat(samples).reset_index(drop=True)

    if len(sampled) < n:
        remaining = df.drop(sampled.index, errors="ignore")
        fill_n = min(n - len(sampled), len(remaining))
        if fill_n > 0:
            fill = remaining.sample(n=fill_n, random_state=seed)
            sampled = pd.concat([sampled, fill]).reset_index(drop=True)

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
