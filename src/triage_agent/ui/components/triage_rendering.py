"""Visualization helpers for a single triage pipeline run."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from triage_agent.schemas import TopicResult, TriageResult, UrgencyResult

ACTION_COLORS = {
    "FORWARD": "#3b82f6",
    "ESCALATE": "#ef4444",
    "CLARIFY": "#f59e0b",
    "FAQ": "#10b981",
    "CLAIM": "#a855f7",
}


def render_topic_section(topic: TopicResult) -> None:
    """Show predicted topic + margin + per-topic scores."""
    st.subheader("Topic Classification")

    col_label, col_margin = st.columns(2)
    with col_label:
        st.metric("Predicted Topic", topic.topic)
    with col_margin:
        margin_label = "near coin-flip" if topic.margin < 0.05 else "confident"
        st.metric(
            "Margin (top1 - top2)",
            f"{topic.margin:+.3f}",
            help=f"Confidence indicator: {margin_label}",
        )

    sorted_scores = sorted(
        topic.all_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    labels = [t for t, _ in sorted_scores]
    scores = [s for _, s in sorted_scores]
    colors = [
        "#3b82f6" if label == topic.topic else "#d1d5db" for label in labels
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=20, t=0, b=0),
        xaxis_title="Cosine similarity",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_urgency_section(urgency: UrgencyResult) -> None:
    """Show urgency level + hybrid score + matched keyword signals."""
    st.subheader("Urgency Classification")

    col_level, col_score = st.columns(2)
    with col_level:
        emoji = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e2"}
        st.metric(
            "Urgency Level",
            f"{emoji.get(urgency.level, '')} {urgency.level}",
        )
    with col_score:
        st.metric("Hybrid Score", f"{urgency.score:.2f}")

    if urgency.signals_found:
        st.markdown(
            "**Matched keyword signals:** "
            + ", ".join(f"`{s}`" for s in urgency.signals_found)
        )
    else:
        st.caption(
            "No explicit urgency keywords matched. Score comes purely from "
            "zero-shot classification."
        )


def render_loop_trace(tools_used: list[str]) -> None:
    """Visualize the tool-use trace as a flow chain."""
    st.subheader("Agentic Loop Trace")

    if not tools_used:
        st.caption("(no tools recorded)")
        return

    chain = " → ".join(f"`{t}`" for t in tools_used)
    st.markdown(f"**Tool path:** {chain}")

    if any(t == "__fallback__" for t in tools_used):
        st.warning(
            "Deterministic fallback fired - the LLM loop did not terminate "
            "cleanly. See orchestrator logs for details."
        )
    if any(t == "__loop_break__" for t in tools_used):
        st.warning(
            "Loop-break guard fired - the LLM tried to call missing_info "
            "twice. Hard guard forced a fallback."
        )

    n_terminal = sum(1 for t in tools_used if t in {"forward", "escalate", "clarify", "faq", "claim"})
    n_helper = sum(1 for t in tools_used if t == "missing_info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total turns", len(tools_used))
    col2.metric("Helper calls", n_helper)
    col3.metric("Terminal calls", n_terminal)


def render_decision_section(result: TriageResult) -> None:
    """Show the final action, next_step, reasoning, and clarification questions."""
    st.subheader("Final Decision")

    color = ACTION_COLORS.get(result.action, "#6b7280")
    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {color};"
        f"background:#f9fafb;border-radius:4px'>"
        f"<strong style='color:{color};font-size:1.2em'>{result.action}</strong>"
        f" → <code>{result.next_step}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**Reasoning:** {result.reasoning}")

    if result.clarification_questions:
        st.markdown("**Clarification questions for the customer:**")
        for q in result.clarification_questions:
            st.markdown(f"- {q}")

    if result.faq_topic:
        st.markdown(f"**FAQ topic:** `{result.faq_topic}`")


def render_runtime(elapsed: float) -> None:
    """Show total triage time as a small footer."""
    st.caption(f"Total pipeline runtime: {elapsed:.2f}s")
