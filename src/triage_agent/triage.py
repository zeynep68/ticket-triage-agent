import json
import logging
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from triage_agent.agent.orchestrator import triage
from triage_agent.logging_config import setup_logging
from triage_agent.schemas import TriageResult

setup_logging()
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    sample: Path = Path("data/sample.parquet"),
    out: Path = Path("data/triage_results.json"),
    n: int | None = None,
    show: bool = True,
) -> None:
    """Run the triage agent on the prepared sample and write results to JSON."""
    if not sample.exists():
        raise typer.BadParameter(
            f"Sample file not found at {sample}. Run `prepare-data` first."
        )

    df = pd.read_parquet(sample)
    if n is not None:
        df = df.head(n)
    log.info("Loaded %d tickets from %s.", len(df), sample)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Triaging"):
        result = triage(text=row["text"], ticket_id=int(row["ticket_id"]))
        ground_truth = {
            "queue": _none_if_nan(row.get("queue")),
            "priority": _none_if_nan(row.get("priority")),
            "type": _none_if_nan(row.get("type")),
        }
        record = result.model_dump()
        record["ground_truth"] = ground_truth
        results.append(record)

        if show:
            _print_comparison(result, ground_truth)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log.info("Wrote %d triage results to %s.", len(results), out)
    action_counts = pd.DataFrame(results)["action"].value_counts().to_dict()
    log.info("Action distribution: %s.", action_counts)


def _none_if_nan(value):
    """Convert pandas NaN to None (JSON can't serialize NaN)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _print_comparison(result: TriageResult, ground_truth: dict) -> None:
    """Print a per-ticket comparison of prediction vs ground truth from the dataset."""
    print(f"\n--- Ticket {result.ticket_id} ---")
    print(f"  Snippet:   {result.text_snippet[:120]}")
    print(
        f"  Predicted: topic={result.topic} | urgency={result.urgency} | "
        f"action={result.action} -> {result.next_step}"
    )
    print(
        f"  Truth:     queue={ground_truth['queue']} | "
        f"priority={ground_truth['priority']} | type={ground_truth['type']}"
    )
    print(f"  Reasoning: {result.reasoning}")
    if result.clarification_questions:
        print(f"  Questions: {result.clarification_questions}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
