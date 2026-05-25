"""Plotly chart helpers for the evaluation dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from triage_agent.schemas import TopicLiteral, UrgencyLiteral

ACTION_ORDER = ["FORWARD", "ESCALATE", "CLARIFY", "FAQ", "CLAIM"]
ACTION_COLORS = {
    "FORWARD": "#3b82f6",
    "ESCALATE": "#ef4444",
    "CLARIFY": "#f59e0b",
    "FAQ": "#10b981",
    "CLAIM": "#a855f7",
}
URGENCY_ORDER: list[UrgencyLiteral] = ["high", "medium", "low"]
TOPIC_ORDER: list[TopicLiteral] = ["Policy", "Claims", "Billing", "Technical", "Other"]


def action_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["action"].value_counts().reindex(ACTION_ORDER, fill_value=0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.55,
                marker=dict(colors=[ACTION_COLORS[a] for a in counts.index]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Action distribution",
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


def tool_path_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    counts = df["tool_path"].value_counts().head(top_n)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.values.tolist(),
                y=counts.index.tolist(),
                orientation="h",
                marker_color="#6366f1",
                text=counts.values.tolist(),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"Tool-path frequency (top {top_n})",
        height=380,
        margin=dict(t=50, b=10, l=10, r=30),
        xaxis_title="Tickets",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def confusion_heatmap(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    order: list[str],
    title: str,
) -> go.Figure:
    subset = df.dropna(subset=[truth_col])
    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)")
        return fig

    matrix = (
        pd.crosstab(subset[truth_col], subset[pred_col])
        .reindex(index=order, columns=order, fill_value=0)
    )
    # Row totals = how many tickets exist per ground-truth class.
    # Column totals = how many tickets the model predicted per class.
    # We embed both into the axis tick labels so the user sees the per-class
    # counts at a glance without computing them from the heatmap cells.
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    y_labels = [f"{label} (n={int(row_totals[label])})" for label in matrix.index]
    x_labels = [f"{label} (n={int(col_totals[label])})" for label in matrix.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=14),
            showscale=False,
        )
    )
    fig.update_layout(
        title=f"{title}  (total: {int(row_totals.sum())})",
        xaxis_title=f"Predicted ({pred_col})",
        yaxis_title=f"Ground truth ({truth_col})",
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def margin_histogram(df: pd.DataFrame) -> go.Figure:
    margins = df["topic_margin"].dropna()
    fig = px.histogram(
        margins,
        nbins=30,
        title="Embedding margin distribution (top-1 - top-2)",
        labels={"value": "Margin"},
    )
    # Matches TOPIC_MARGIN_THRESHOLD in agent/tools/topic.py: below this the
    # classifier falls back to "Other" because top-1 and top-2 are too close.
    fig.add_vline(
        x=0.01,
        line_dash="dash",
        line_color="red",
        annotation_text="margin < 0.01 = weak separation (Other fallback)",
        annotation_position="top right",
    )
    fig.update_layout(
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


def runtime_boxplot(df: pd.DataFrame) -> go.Figure:
    df_t = df.dropna(subset=["runtime_seconds"]).copy()
    if df_t.empty:
        fig = go.Figure()
        fig.update_layout(title="Runtime per tool-path (no data)")
        return fig

    order = (
        df_t.groupby("tool_path")["runtime_seconds"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fig = px.box(
        df_t,
        x="runtime_seconds",
        y="tool_path",
        category_orders={"tool_path": order},
        title="Runtime per tool-path (seconds)",
        labels={"runtime_seconds": "Seconds"},
    )
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=10, l=10, r=10),
        yaxis_title="",
    )
    return fig


def length_action_stacked(df: pd.DataFrame) -> go.Figure:
    df_l = df.dropna(subset=["text_length"]).copy()
    if df_l.empty:
        fig = go.Figure()
        fig.update_layout(title="Length bucket vs action (no data)")
        return fig

    q25, q75 = df_l["text_length"].quantile(0.25), df_l["text_length"].quantile(0.75)
    df_l["length_bucket"] = pd.cut(
        df_l["text_length"],
        bins=[-1, q25, q75, float("inf")],
        labels=[f"short (<={int(q25)})", f"medium ({int(q25) + 1}-{int(q75)})", f"long (>{int(q75)})"],
    )

    counts = pd.crosstab(df_l["length_bucket"], df_l["action"])
    counts = counts.reindex(columns=[a for a in ACTION_ORDER if a in counts.columns], fill_value=0)

    fig = go.Figure()
    for action in counts.columns:
        fig.add_trace(
            go.Bar(
                name=action,
                x=counts.index.astype(str).tolist(),
                y=counts[action].tolist(),
                marker_color=ACTION_COLORS.get(action, "#6b7280"),
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Action mix per length bucket",
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        xaxis_title="",
        yaxis_title="Tickets",
    )
    return fig
