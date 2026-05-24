"""Evaluate triage_results.json against the dataset's queue and priority labels.

Computes agreement metrics, action and tool-path distributions, confusion
matrices, and sample disagreement cases. Produces a readable report intended
for the technical writeup.

Usage:
    uv run python -m triage_agent.evaluation.run_eval
    uv run python -m triage_agent.evaluation.run_eval --results data/triage_results.json
"""

import json
import logging
from pathlib import Path
from typing import get_args

import pandas as pd
import typer

from triage_agent.evaluation.ground_truth import (
    map_priority_to_urgency,
    map_queue_to_topic,
)
from triage_agent.logging_config import setup_logging
from triage_agent.schemas import TopicLiteral, UrgencyLiteral

setup_logging()
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

URGENCY_NUMERIC: dict[UrgencyLiteral, int] = {"low": 0, "medium": 1, "high": 2}
URGENCY_ORDER: list[UrgencyLiteral] = ["high", "medium", "low"]  # display order
TOPIC_ORDER: list[TopicLiteral] = list(get_args(TopicLiteral))


def load_results(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a per-ticket DataFrame with both predictions and mapped ground truth."""
    rows = []
    for r in results:
        gt = r.get("ground_truth", {}) or {}
        clarifications = r.get("clarification_questions") or []
        rows.append(
            {
                "ticket_id": r["ticket_id"],
                "predicted_topic": r["topic"],
                "predicted_urgency": r["urgency"],
                "topic_margin": r.get("topic_margin"),
                "action": r["action"],
                "next_step": r["next_step"],
                "tools_used": tuple(r.get("tools_used", [])),
                "tool_path": " -> ".join(r.get("tools_used", [])) or "(none)",
                "n_clarifications": len(clarifications),
                "runtime_seconds": r.get("runtime_seconds"),
                "text_length": r.get("text_length"),
                "gt_queue": gt.get("queue"),
                "gt_priority": gt.get("priority"),
                "mapped_topic": map_queue_to_topic(gt.get("queue")),
                "mapped_urgency": map_priority_to_urgency(gt.get("priority")),
                "snippet": r.get("text_snippet", ""),
                "reasoning": r.get("reasoning", ""),
            }
        )
    return pd.DataFrame(rows)


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_overall(df: pd.DataFrame) -> None:
    _section("OVERALL")
    print(f"\nTotal tickets evaluated: {len(df):,}")

    print("\nAction distribution:")
    counts = df["action"].value_counts()
    for action, n in counts.items():
        pct = n / len(df) * 100
        print(f"  {action:<10}  {n:>5}  ({pct:5.1f}%)")

    print("\nTool-path distribution (top 10):")
    path_counts = df["tool_path"].value_counts()
    for path, n in path_counts.head(10).items():
        pct = n / len(df) * 100
        print(f"  {n:>5}  ({pct:5.1f}%)  {path}")
    if len(path_counts) > 10:
        print(f"  ... and {len(path_counts) - 10} more distinct paths")

    fallback_count = sum(1 for tools in df["tools_used"] if "__fallback__" in tools)
    if fallback_count:
        fb_pct = fallback_count / len(df) * 100
        print(
            f"\nFallback rate: {fallback_count}/{len(df)} ({fb_pct:.1f}%) "
            f"- tickets where the LLM loop failed and the deterministic fallback fired."
        )
    else:
        print("\nFallback rate: 0 (no LLM-loop failures)")

    print("\nAction x tool-path crosstab:")
    crosstab = pd.crosstab(df["action"], df["tool_path"], margins=True, margins_name="Total")
    print(crosstab.to_string())

    print("\nClarification-question coverage (orthogonal to routing):")
    for action in ["FORWARD", "ESCALATE", "CLARIFY", "FAQ"]:
        bucket = df[df["action"] == action]
        if bucket.empty:
            continue
        with_q = (bucket["n_clarifications"] > 0).sum()
        avg_q = bucket["n_clarifications"].mean()
        pct = with_q / len(bucket) * 100
        print(
            f"  {action:<10}  {with_q:>3} / {len(bucket):<3} with questions "
            f"({pct:5.1f}%)  avg={avg_q:.2f} per ticket"
        )


def report_urgency(df: pd.DataFrame) -> None:
    _section("URGENCY (predicted vs dataset priority -> mapped urgency)")

    df_u = df.dropna(subset=["mapped_urgency"])
    if df_u.empty:
        print("\nNo tickets with mappable priority - skipping urgency evaluation.")
        return

    print(f"\nComparable tickets: {len(df_u):,} / {len(df):,}")
    print(
        f"({len(df) - len(df_u):,} tickets had no mappable priority and are excluded.)"
    )

    exact = (df_u["predicted_urgency"] == df_u["mapped_urgency"]).sum()
    exact_pct = exact / len(df_u) * 100

    df_u = df_u.copy()
    df_u["pred_num"] = df_u["predicted_urgency"].map(URGENCY_NUMERIC)
    df_u["gt_num"] = df_u["mapped_urgency"].map(URGENCY_NUMERIC)
    one_off = (df_u["pred_num"] - df_u["gt_num"]).abs().le(1).sum()
    one_off_pct = one_off / len(df_u) * 100

    print(f"\n  Exact match:       {exact:>5} / {len(df_u):<5}  ({exact_pct:5.1f}%)")
    print(f"  Within +/- 1 level: {one_off:>5} / {len(df_u):<5}  ({one_off_pct:5.1f}%)")

    print("\nPer-level breakdown (mapped_urgency on rows, predicted on columns):")
    matrix = pd.crosstab(
        df_u["mapped_urgency"],
        df_u["predicted_urgency"],
        margins=True,
        margins_name="Total",
    )
    matrix = matrix.reindex(
        index=[*URGENCY_ORDER, "Total"],
        columns=[*URGENCY_ORDER, "Total"],
        fill_value=0,
    )
    print(matrix.to_string())


def report_topic(df: pd.DataFrame) -> None:
    _section("TOPIC (predicted vs dataset queue -> mapped topic)")

    df_t = df.dropna(subset=["mapped_topic"])
    if df_t.empty:
        print("\nNo tickets with mappable queue - skipping topic evaluation.")
        return

    print(f"\nComparable tickets: {len(df_t):,} / {len(df):,}")
    print(
        "(All tickets contribute; queues without an explicit mapping count as 'Other'.)"
    )

    exact = (df_t["predicted_topic"] == df_t["mapped_topic"]).sum()
    exact_pct = exact / len(df_t) * 100
    print(f"\n  Topic agreement (all):           {exact:>3} / {len(df_t):<3}  ({exact_pct:5.1f}%)")

    # Excluding "Other" - more honest signal because most "Other" mappings
    # come from queues without a semantic mapping (organisational, not topical).
    df_t_mapped = df_t[df_t["mapped_topic"] != "Other"]
    if not df_t_mapped.empty:
        exact_mapped = (
            df_t_mapped["predicted_topic"] == df_t_mapped["mapped_topic"]
        ).sum()
        pct_mapped = exact_mapped / len(df_t_mapped) * 100
        print(
            f"  Topic agreement (excl. Other):   "
            f"{exact_mapped:>3} / {len(df_t_mapped):<3}  ({pct_mapped:5.1f}%)"
        )

    # Embedding-margin statistics: low margin indicates uncertain classification.
    margins = df_t["topic_margin"].dropna()
    if not margins.empty:
        low_margin = (margins < 0.05).sum()
        print("\nEmbedding-margin stats (top1 minus top2 cosine similarity):")
        print(
            f"  Average margin:           {margins.mean():.3f}  "
            f"(min={margins.min():.3f}, max={margins.max():.3f})"
        )
        print(
            f"  Low-margin tickets (< 0.05): {low_margin} / {len(margins)} "
            f"({low_margin / len(margins) * 100:.1f}%) - near coin-flip "
            f"between top-1 and top-2 topic."
        )

    print("\nConfusion matrix (mapped_topic on rows, predicted on columns):")
    matrix = pd.crosstab(
        df_t["mapped_topic"],
        df_t["predicted_topic"],
        margins=True,
        margins_name="Total",
    )
    matrix = matrix.reindex(
        index=[*TOPIC_ORDER, "Total"],
        columns=[*TOPIC_ORDER, "Total"],
        fill_value=0,
    )
    print(matrix.to_string())

    print("\nPer-topic agreement (predicted-topic correctness on its own bucket):")
    print(f"  {'topic':<12}  {'tickets':>8}  {'correct':>8}  {'pct':>7}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 7}")
    for topic in TOPIC_ORDER:
        bucket = df_t[df_t["mapped_topic"] == topic]
        n = len(bucket)
        if n == 0:
            continue
        correct = (bucket["predicted_topic"] == topic).sum()
        pct = correct / n * 100
        print(f"  {topic:<12}  {n:>8}  {correct:>8}  {pct:>6.1f}%")


def report_length_distribution(df: pd.DataFrame) -> None:
    _section("TICKET LENGTH (characters: subject + body)")

    lengths = df["text_length"].dropna()
    if lengths.empty:
        print(
            "\nNo text_length info in results - re-run triage to capture it."
        )
        return

    print("\nLength quantiles:")
    print(f"  min:     {int(lengths.min()):>5}")
    print(f"  p25:     {int(lengths.quantile(0.25)):>5}")
    print(f"  median:  {int(lengths.median()):>5}")
    print(f"  p75:     {int(lengths.quantile(0.75)):>5}")
    print(f"  p95:     {int(lengths.quantile(0.95)):>5}")
    print(f"  max:     {int(lengths.max()):>5}")

    print("\nPer-action mean length:")
    for action, bucket in df.groupby("action"):
        mean_len = bucket["text_length"].mean()
        print(f"  {action:<10}  {int(mean_len):>5} chars  (n={len(bucket)})")

    # Bucket the tickets into short / medium / long using the dataset's own
    # quartiles so the bins are data-driven, not arbitrary.
    q25, q75 = lengths.quantile(0.25), lengths.quantile(0.75)
    df_l = df.dropna(subset=["text_length"]).copy()
    df_l["length_bucket"] = pd.cut(
        df_l["text_length"],
        bins=[-1, q25, q75, float("inf")],
        labels=[f"short (<={int(q25)})", f"medium ({int(q25) + 1}-{int(q75)})", f"long (>{int(q75)})"],
    )

    print("\nLength bucket vs action:")
    crosstab = pd.crosstab(
        df_l["length_bucket"], df_l["action"], margins=True, margins_name="Total"
    )
    print(crosstab.to_string())

    # Quality breakdown per bucket: did short/long tickets get more agreement?
    df_t = df_l.dropna(subset=["mapped_topic"])
    df_u = df_l.dropna(subset=["mapped_urgency"])
    if not df_t.empty or not df_u.empty:
        print("\nAgreement quality per length bucket:")
        print(f"  {'bucket':<22}  {'topic agree':>11}  {'urgency exact':>13}  {'avg margin':>10}")
        print(f"  {'-' * 22}  {'-' * 11}  {'-' * 13}  {'-' * 10}")
        for bucket_label in df_l["length_bucket"].cat.categories:
            bucket_t = df_t[df_t["length_bucket"] == bucket_label]
            bucket_u = df_u[df_u["length_bucket"] == bucket_label]
            n_t = len(bucket_t)
            n_u = len(bucket_u)

            if n_t > 0:
                topic_pct = (
                    (bucket_t["predicted_topic"] == bucket_t["mapped_topic"]).sum()
                    / n_t * 100
                )
                topic_str = f"{topic_pct:5.1f}% ({n_t})"
            else:
                topic_str = "  - / -    "

            if n_u > 0:
                urg_pct = (
                    (bucket_u["predicted_urgency"] == bucket_u["mapped_urgency"]).sum()
                    / n_u * 100
                )
                urg_str = f"{urg_pct:5.1f}% ({n_u})"
            else:
                urg_str = "  - / -      "

            margins = df_l[df_l["length_bucket"] == bucket_label]["topic_margin"].dropna()
            margin_str = f"{margins.mean():.3f}" if not margins.empty else "  -  "

            print(f"  {bucket_label:<22}  {topic_str:>11}  {urg_str:>13}  {margin_str:>10}")


def report_performance(df: pd.DataFrame) -> None:
    _section("PERFORMANCE")

    runtimes = df["runtime_seconds"].dropna()
    if runtimes.empty:
        print(
            "\nNo runtime info in results - re-run triage to capture per-ticket timing."
        )
        return

    total = runtimes.sum()
    avg = runtimes.mean()
    p50 = runtimes.median()
    p95 = runtimes.quantile(0.95)
    print(
        f"\nTotal runtime:           {total:7.2f}s across {len(runtimes)} tickets"
    )
    print(f"Average per ticket:      {avg:7.2f}s")
    print(f"Median (p50):            {p50:7.2f}s")
    print(f"p95:                     {p95:7.2f}s")
    print(f"Slowest ticket:          {runtimes.max():7.2f}s")

    # Runtime per tool-path - helps see where the loop costs latency.
    print("\nAverage runtime per tool-path:")
    by_path = df.dropna(subset=["runtime_seconds"]).groupby("tool_path")[
        "runtime_seconds"
    ].agg(["count", "mean"])
    by_path = by_path.sort_values("mean", ascending=False)
    for path, row in by_path.iterrows():
        print(f"  {row['mean']:5.2f}s  (n={int(row['count']):>2})  {path}")


def report_disagreements(df: pd.DataFrame, n: int = 5) -> None:
    _section(f"SAMPLE DISAGREEMENTS (up to {n} per category)")

    df_t = df.dropna(subset=["mapped_topic"])
    topic_mismatch = df_t[df_t["predicted_topic"] != df_t["mapped_topic"]]

    df_u = df.dropna(subset=["mapped_urgency"])
    urgency_mismatch = df_u[df_u["predicted_urgency"] != df_u["mapped_urgency"]]

    print(f"\nTopic disagreements: {len(topic_mismatch):,}")
    print(f"Urgency disagreements: {len(urgency_mismatch):,}")

    print(f"\n--- Topic disagreement examples (up to {n}) ---")
    for _, row in topic_mismatch.head(n).iterrows():
        _print_example_case(row, focus="topic")

    print(f"\n--- Urgency disagreement examples (up to {n}) ---")
    for _, row in urgency_mismatch.head(n).iterrows():
        _print_example_case(row, focus="urgency")


def _print_example_case(row: pd.Series, focus: str) -> None:
    print()
    print(f"  Ticket {row['ticket_id']}")
    print(f"    Snippet:   {row['snippet'][:140]}")
    if focus == "topic":
        print(
            f"    Predicted topic:  {row['predicted_topic']}  | "
            f"Ground-truth mapped: {row['mapped_topic']}  (queue={row['gt_queue']!r})"
        )
    elif focus == "urgency":
        print(
            f"    Predicted urgency: {row['predicted_urgency']}  | "
            f"Ground-truth mapped: {row['mapped_urgency']}  (priority={row['gt_priority']!r})"
        )
    print(f"    Action:    {row['action']} (via {' -> '.join(row['tools_used'])})")
    if row["reasoning"]:
        print(f"    Reasoning: {row['reasoning'][:140]}")


@app.command()
def evaluate(results: Path = Path("data/triage_results.json")) -> None:
    """Read the triage results file and print an evaluation report."""
    if not results.exists():
        raise typer.BadParameter(
            f"Results file not found at {results}. Run `triage` first."
        )

    raw = load_results(results)
    df = build_dataframe(raw)

    report_overall(df)
    report_urgency(df)
    report_topic(df)
    report_length_distribution(df)
    report_performance(df)
    report_disagreements(df)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
