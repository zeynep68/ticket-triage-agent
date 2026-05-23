"""Inspect the dataset's label columns (queue, priority) and how they map
onto our taxonomies.

Run this after `prepare-data` to:
  - see every distinct queue and priority value in the data
  - see how each maps (or doesn't) to our 6-topic / 3-urgency taxonomies
  - get a copy-paste skeleton for any unmapped values that need attention

Usage:
    uv run python -m triage_agent.evaluation.inspect_labels
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import get_args

import pandas as pd
import typer

from triage_agent.evaluation.ground_truth import (
    QUEUE_TO_TOPIC,
    map_priority_to_urgency,
    map_queue_to_topic,
)
from triage_agent.logging_config import setup_logging
from triage_agent.schemas import TopicLiteral

setup_logging()
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# Display order - derived from the TopicLiteral type so adding a topic in
# schemas.py automatically extends the inspection output.
TOPIC_ORDER = list(get_args(TopicLiteral))
URGENCY_ORDER = ["high", "medium", "low"]  # display order, severe to mild


@app.command()
def inspect(sample: Path = Path("data/sample.parquet")) -> None:
    """Show unique queue and priority values with mapping summary."""
    if not sample.exists():
        raise typer.BadParameter(
            f"Sample file not found at {sample}. Run `prepare-data` first."
        )

    df = pd.read_parquet(sample)
    _inspect_priorities(df)
    _inspect_queues(df)


def _section_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _inspect_priorities(df: pd.DataFrame) -> None:
    _section_header("PRIORITIES IN DATA -> MAPPED URGENCY")

    if "priority" not in df.columns:
        log.warning("No 'priority' column in the sample.")
        return

    counts = df["priority"].value_counts(dropna=False)
    print(f"\n{len(counts)} distinct priority values across {len(df):,} tickets.\n")

    # Per-value detail
    print(f"  {'dataset value':<14}  ->  {'mapped to':<10}   tickets")
    print(f"  {'-' * 14}  --  {'-' * 10}   {'-' * 7}")

    unmapped = []
    summary: dict[str, int] = defaultdict(int)

    for value, count in counts.items():
        if pd.isna(value):
            mapped = None
            label = "(missing)"
        else:
            mapped = map_priority_to_urgency(value)
            label = "UNMAPPED" if mapped is None else mapped

        if mapped is None and not pd.isna(value):
            unmapped.append((value, count))
        elif mapped is not None:
            summary[mapped] += count

        display_value = "(NaN)" if pd.isna(value) else str(value)
        print(f"  {display_value:<14}  ->  {label:<10}   {count:>7,}")

    # Summary by mapped urgency
    print("\n  After mapping, tickets per urgency level:")
    total = sum(summary.values())
    for urg in URGENCY_ORDER:
        n = summary.get(urg, 0)
        pct = (n / total * 100) if total else 0
        print(f"    {urg:<8}  {n:>7,}  ({pct:5.1f}%)")

    if unmapped:
        print("\n  Unmapped priority values (need a rule in map_priority_to_urgency):")
        for value, count in unmapped:
            print(f"    {value!r:<20}  {count:>7,} tickets")


def _inspect_queues(df: pd.DataFrame) -> None:
    _section_header("QUEUES IN DATA -> MAPPED TOPIC")

    if "queue" not in df.columns:
        log.warning("No 'queue' column in the sample.")
        return

    counts = df["queue"].value_counts(dropna=False)
    print(f"\n{len(counts)} distinct queue values across {len(df):,} tickets.")
    print(
        f"{len(QUEUE_TO_TOPIC)} queues have explicit mappings; the rest default to 'Other'.\n"
    )

    # Group queues by their mapped topic
    by_topic: dict[str, list[tuple[str, int]]] = defaultdict(list)
    unmapped_to_other: list[tuple[str, int]] = []
    explicit_keys = set(QUEUE_TO_TOPIC.keys())

    for queue, count in counts.items():
        if pd.isna(queue):
            continue
        mapped = map_queue_to_topic(queue)
        if mapped is None:
            continue
        by_topic[mapped].append((queue, count))
        if queue not in explicit_keys:
            unmapped_to_other.append((queue, count))

    # Print one block per topic, in canonical order
    for topic in TOPIC_ORDER:
        entries = by_topic.get(topic, [])
        if not entries:
            continue
        total = sum(c for _, c in entries)
        print(f"-- {topic}  ({total:,} tickets, {len(entries)} queues)")
        for queue, count in entries:
            marker = " " if queue in explicit_keys else "*"
            print(f"   {marker} {count:>6,}  {queue}")
        print("-" * 60)
        print()

    # Summary by topic
    print("Topic distribution summary:")
    grand_total = sum(sum(c for _, c in entries) for entries in by_topic.values())
    for topic in TOPIC_ORDER:
        entries = by_topic.get(topic, [])
        n = sum(c for _, c in entries)
        pct = (n / grand_total * 100) if grand_total else 0
        marker = "" if n > 0 else "  (none)"
        print(f"  {topic:<10}  {n:>7,}  ({pct:5.1f}%){marker}")

    print("\nLegend:  *  queue falls through to 'Other' default (no explicit mapping)")

    if unmapped_to_other:
        print(f"\n{len(unmapped_to_other)} queues currently default to 'Other'.")
        print("If any of these should map to a specific topic, add them to")
        print("QUEUE_TO_TOPIC in src/triage_agent/evaluation/ground_truth.py:\n")
        for queue, count in unmapped_to_other[:10]:
            print(f'    "{queue}": "Other",  # {count:,} tickets')
        if len(unmapped_to_other) > 10:
            print(f"    ... and {len(unmapped_to_other) - 10} more")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
