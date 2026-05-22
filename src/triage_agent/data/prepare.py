import logging
from pathlib import Path

import kagglehub
import typer

from triage_agent.data.constants import TARGET_LANGUAGES
from triage_agent.data.loader import load_all_csvs
from triage_agent.data.preprocess import apply_preprocessing, filter_languages
from triage_agent.data.sampling import stratified_sample
from triage_agent.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def prepare(
    out: Path = Path("data/sample.parquet"),
    rejected_out: Path = Path("data/rejected.csv"),
    n_samples: int = 5_000,
    seed: int = 42,
) -> None:
    """Download, validate, clean, and sample the Kaggle dataset."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets"))
    log.info("Downloaded the Kaggle dataset to %s.", raw_dir)

    valid_df, rejected_df = load_all_csvs(raw_dir)
    log.info(
        "Loaded %d valid tickets and %d rejected rows across all CSVs.",
        len(valid_df),
        len(rejected_df),
    )

    rejected_df.to_csv(rejected_out, index=False)
    log.info("Wrote rejected rows to %s for auditing.", rejected_out)

    valid_df = filter_languages(valid_df, TARGET_LANGUAGES)
    log.info("Filtered to target languages, %d tickets remain.", len(valid_df))

    valid_df = apply_preprocessing(valid_df)
    log.info("Applied text preprocessing, %d tickets remain.", len(valid_df))

    sample_df = stratified_sample(valid_df, n=n_samples, stratify_by="queue", seed=seed)
    sample_df.to_parquet(out, index=False)
    log.info("Wrote final sample of %d tickets to %s.", len(sample_df), out)

    log.info(
        "Language distribution in the sample: %s.",
        sample_df["language"].value_counts().to_dict(),
    )
    log.info(
        "Version distribution in the sample: %s.",
        sample_df["version"].value_counts().to_dict(),
    )
    if "queue" in sample_df.columns:
        log.info(
            "Top 10 queues in the sample: %s.",
            sample_df["queue"].value_counts().head(10).to_dict(),
        )
    log.info(
        "Text length statistics: mean=%.0f characters, median=%.0f, max=%d.",
        sample_df["text_length"].mean(),
        sample_df["text_length"].median(),
        sample_df["text_length"].max(),
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
