"""Evaluation dashboard: load triage_results.json and visualize aggregate stats."""

from pathlib import Path

import pandas as pd
import streamlit as st

from triage_agent.evaluation.run_eval import build_dataframe, load_results
from triage_agent.ui.components.eval_rendering import (
    TOPIC_ORDER,
    URGENCY_ORDER,
    action_donut,
    confusion_heatmap,
    length_action_stacked,
    margin_histogram,
    runtime_boxplot,
    tool_path_bar,
)

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="\U0001f4ca",
    layout="wide",
)

st.title("Evaluation Dashboard")
st.caption(
    "Aggregate metrics over the triage results file. Mirrors `run_eval.py` "
    "with interactive plotly charts."
)

DEFAULT_PATH = Path("data/triage_results.json")


@st.cache_data(show_spinner=False)
def _load_dataframe(path_str: str, mtime: float) -> pd.DataFrame:
    """Cached by (path, mtime) so the dataframe rebuilds only when file changes."""
    raw = load_results(Path(path_str))
    return build_dataframe(raw)


with st.sidebar:
    st.header("Results file")
    path_input = st.text_input(
        "Path to triage_results.json",
        value=str(DEFAULT_PATH),
    )
    results_path = Path(path_input)

    if not results_path.exists():
        st.error(f"File not found: {results_path}")
        st.stop()

    df = _load_dataframe(str(results_path), results_path.stat().st_mtime)
    st.success(f"Loaded {len(df):,} tickets")

    st.markdown("---")
    st.header("Filters")
    action_filter = st.multiselect(
        "Actions",
        options=sorted(df["action"].unique().tolist()),
        default=sorted(df["action"].unique().tolist()),
    )
    topic_filter = st.multiselect(
        "Predicted topics",
        options=sorted(df["predicted_topic"].unique().tolist()),
        default=sorted(df["predicted_topic"].unique().tolist()),
    )
    urgency_filter = st.multiselect(
        "Predicted urgency",
        options=URGENCY_ORDER,
        default=URGENCY_ORDER,
    )

df_filtered = df[
    df["action"].isin(action_filter)
    & df["predicted_topic"].isin(topic_filter)
    & df["predicted_urgency"].isin(urgency_filter)
]


# Headline KPIs
st.subheader("Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tickets (filtered)", f"{len(df_filtered):,}")

topic_subset = df_filtered.dropna(subset=["mapped_topic"])
if not topic_subset.empty:
    topic_agree = (
        (topic_subset["predicted_topic"] == topic_subset["mapped_topic"]).sum()
        / len(topic_subset) * 100
    )
    col2.metric("Topic agreement", f"{topic_agree:.1f}%")

    excl_other = topic_subset[topic_subset["mapped_topic"] != "Other"]
    if not excl_other.empty:
        agree_excl = (
            (excl_other["predicted_topic"] == excl_other["mapped_topic"]).sum()
            / len(excl_other) * 100
        )
        col3.metric("Topic agreement (excl Other)", f"{agree_excl:.1f}%")

urg_subset = df_filtered.dropna(subset=["mapped_urgency"])
if not urg_subset.empty:
    urg_exact = (
        (urg_subset["predicted_urgency"] == urg_subset["mapped_urgency"]).sum()
        / len(urg_subset) * 100
    )
    col4.metric("Urgency exact", f"{urg_exact:.1f}%")

fallback_count = sum(1 for t in df_filtered["tools_used"] if "__fallback__" in t)
col5.metric("Fallback rate", f"{fallback_count}/{len(df_filtered)}")


st.markdown("---")

# Distribution row
st.subheader("Distributions")
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(action_donut(df_filtered), use_container_width=True)
with col_b:
    st.plotly_chart(tool_path_bar(df_filtered), use_container_width=True)

# Confusion row
st.subheader("Confusion matrices")
col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(
        confusion_heatmap(
            df_filtered,
            truth_col="mapped_topic",
            pred_col="predicted_topic",
            order=TOPIC_ORDER,
            title="Topic: ground truth vs prediction",
        ),
        use_container_width=True,
    )
with col_d:
    st.plotly_chart(
        confusion_heatmap(
            df_filtered,
            truth_col="mapped_urgency",
            pred_col="predicted_urgency",
            order=URGENCY_ORDER,
            title="Urgency: ground truth vs prediction",
        ),
        use_container_width=True,
    )

# Embedding margin + length
st.subheader("Embedding confidence and ticket length")
col_e, col_f = st.columns(2)
with col_e:
    st.plotly_chart(margin_histogram(df_filtered), use_container_width=True)
with col_f:
    st.plotly_chart(length_action_stacked(df_filtered), use_container_width=True)

# Performance
st.subheader("Performance")
st.plotly_chart(runtime_boxplot(df_filtered), use_container_width=True)


st.markdown("---")

# Sample disagreements
st.subheader("Sample disagreements")

tab_topic, tab_urgency = st.tabs(["Topic", "Urgency"])

with tab_topic:
    topic_mismatch = topic_subset[
        topic_subset["predicted_topic"] != topic_subset["mapped_topic"]
    ]
    st.caption(f"{len(topic_mismatch)} topic disagreements (filtered)")
    if not topic_mismatch.empty:
        for _, row in topic_mismatch.head(10).iterrows():
            with st.expander(
                f"Ticket {row['ticket_id']}: "
                f"{row['predicted_topic']} vs {row['mapped_topic']} "
                f"(queue={row['gt_queue']!r})"
            ):
                st.markdown(f"**Snippet:** {row['snippet']}")
                st.markdown(f"**Action:** {row['action']} (via {' -> '.join(row['tools_used'])})")
                if row["reasoning"]:
                    st.markdown(f"**Reasoning:** {row['reasoning']}")

with tab_urgency:
    urg_mismatch = urg_subset[
        urg_subset["predicted_urgency"] != urg_subset["mapped_urgency"]
    ]
    st.caption(f"{len(urg_mismatch)} urgency disagreements (filtered)")
    if not urg_mismatch.empty:
        for _, row in urg_mismatch.head(10).iterrows():
            with st.expander(
                f"Ticket {row['ticket_id']}: "
                f"{row['predicted_urgency']} vs {row['mapped_urgency']} "
                f"(priority={row['gt_priority']!r})"
            ):
                st.markdown(f"**Snippet:** {row['snippet']}")
                st.markdown(f"**Action:** {row['action']} (via {' -> '.join(row['tools_used'])})")
                if row["reasoning"]:
                    st.markdown(f"**Reasoning:** {row['reasoning']}")

# Raw data viewer
with st.expander("Raw filtered dataframe"):
    st.dataframe(df_filtered, use_container_width=True)
