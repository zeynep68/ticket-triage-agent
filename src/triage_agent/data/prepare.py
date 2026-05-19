import logging
from pathlib import Path

import kagglehub
import typer

from triage_agent.data.loader import load_all_csvs
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

    raw_dir = Path(
        kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")
    )
    log.info("Downloaded the Kaggle dataset to %s.", raw_dir)

    valid_df, rejected_df = load_all_csvs(raw_dir)
    log.info(
        "Loaded %d valid tickets and %d rejected rows across all CSVs.",
        len(valid_df),
        len(rejected_df),
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
