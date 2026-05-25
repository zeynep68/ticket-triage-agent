"""Data statistics page: explore the prepared sample.parquet before triage."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from triage_agent.evaluation.ground_truth import QUEUE_TO_TOPIC

st.set_page_config(
    page_title="Data Statistics",
    page_icon="\U0001f4cf",
    layout="wide",
)

st.title("Data Statistics")
st.caption(
    "Explore the prepared `sample.parquet` dataset before any triage runs: "
    "language mix, queue distribution, priority and length characteristics."
)

DEFAULT_PATH = Path("data/sample.parquet")


@st.cache_data(show_spinner=False)
def _load_sample(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


with st.sidebar:
    st.header("Sample file")
    path_input = st.text_input(
        "Path to sample.parquet",
        value=str(DEFAULT_PATH),
    )
    path = Path(path_input)
    if not path.exists():
        st.error(f"File not found: {path}")
        st.stop()

    df = _load_sample(str(path), path.stat().st_mtime)
    st.success(f"Loaded {len(df):,} tickets")

    st.markdown("---")
    st.header("Filter")
    languages = sorted(df["language"].dropna().unique().tolist())
    lang_filter = st.multiselect(
        "Languages",
        options=languages,
        default=languages,
    )

df_filtered = df[df["language"].isin(lang_filter)]


# Overview KPIs
st.subheader("Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tickets (filtered)", f"{len(df_filtered):,}")
col2.metric("Languages", df_filtered["language"].nunique())
col3.metric("Distinct queues", df_filtered["queue"].nunique())
col4.metric("Distinct priorities", df_filtered["priority"].nunique())
col5.metric("Distinct types", df_filtered["type"].nunique())


st.markdown("---")

# Language + Priority distributions side by side
st.subheader("Language and priority distribution")
col_lang, col_prio = st.columns(2)
with col_lang:
    lang_counts = df_filtered["language"].value_counts()
    fig_lang = px.pie(
        values=lang_counts.values,
        names=lang_counts.index,
        title="Tickets per language",
        hole=0.4,
    )
    fig_lang.update_traces(textinfo="label+percent")
    fig_lang.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_lang, use_container_width=True)

with col_prio:
    prio_order = ["very_low", "low", "medium", "high", "critical"]
    prio_counts = df_filtered["priority"].value_counts()
    # Reorder if known priority levels exist
    ordered_index = [p for p in prio_order if p in prio_counts.index] + [
        p for p in prio_counts.index if p not in prio_order
    ]
    prio_counts = prio_counts.reindex(ordered_index)
    fig_prio = px.bar(
        x=prio_counts.index.astype(str).tolist(),
        y=prio_counts.values.tolist(),
        title="Tickets per priority",
        labels={"x": "Priority", "y": "Tickets"},
        color=prio_counts.index.astype(str).tolist(),
        color_discrete_sequence=px.colors.sequential.Reds,
    )
    fig_prio.update_layout(
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_prio, use_container_width=True)


# Queue distribution
st.subheader("Top queues")
queue_counts = df_filtered["queue"].value_counts().head(20)
fig_queue = px.bar(
    x=queue_counts.values.tolist(),
    y=queue_counts.index.tolist(),
    orientation="h",
    title=f"Top {len(queue_counts)} queues by ticket count",
    labels={"x": "Tickets", "y": "Queue"},
)
fig_queue.update_layout(
    height=500,
    margin=dict(t=50, b=10, l=10, r=30),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_queue, use_container_width=True)


# Queue → topic mapping coverage
st.subheader("Queue-to-topic mapping coverage")
df_map = df_filtered.copy()
df_map["mapped_topic"] = df_map["queue"].map(
    lambda q: QUEUE_TO_TOPIC.get(q, "Other") if isinstance(q, str) else None
)
mapping_counts = df_map["mapped_topic"].value_counts()
col_map, col_explain = st.columns([1, 1])
with col_map:
    fig_map = px.pie(
        values=mapping_counts.values,
        names=mapping_counts.index,
        title="Tickets per mapped topic (insurance taxonomy)",
        hole=0.4,
    )
    fig_map.update_traces(textinfo="label+percent")
    fig_map.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_map, use_container_width=True)
with col_explain:
    st.markdown(
        f"""
        Out of **{df_map['queue'].nunique()} distinct queues**, only
        **{len([q for q in df_map['queue'].unique() if q in QUEUE_TO_TOPIC])}**
        have an explicit insurance-context mapping
        (Policy / Claims / Billing / Technical). The remaining queues fall
        through to **Other**.

        This is the main reason the Eval Dashboard reports a much higher
        topic-agreement *excl. Other* (~58%) than overall (~31%): a large
        share of mapped-Other tickets simply do not have a clean insurance
        analog in the Kaggle dataset.
        """
    )


# Ticket length
st.subheader("Ticket length distribution")
col_hist, col_box = st.columns(2)

# Length column might be missing if user supplies an older parquet.
if "text_length" not in df_filtered.columns:
    df_filtered = df_filtered.copy()
    df_filtered["text_length"] = df_filtered["text"].astype(str).str.len()

with col_hist:
    fig_len_hist = px.histogram(
        df_filtered,
        x="text_length",
        nbins=40,
        title="Distribution of ticket text length (characters)",
        labels={"text_length": "Length (characters)"},
    )
    fig_len_hist.update_layout(
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_len_hist, use_container_width=True)

with col_box:
    fig_len_box = px.box(
        df_filtered,
        x="language",
        y="text_length",
        title="Ticket length per language",
        labels={"text_length": "Length (characters)", "language": "Language"},
        color="language",
    )
    fig_len_box.update_layout(
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_len_box, use_container_width=True)

length_quantiles = df_filtered["text_length"].describe(
    percentiles=[0.25, 0.5, 0.75, 0.95]
)
qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
qcol1.metric("min", f"{int(length_quantiles['min']):,}")
qcol2.metric("p25", f"{int(length_quantiles['25%']):,}")
qcol3.metric("median", f"{int(length_quantiles['50%']):,}")
qcol4.metric("p75", f"{int(length_quantiles['75%']):,}")
qcol5.metric("max", f"{int(length_quantiles['max']):,}")


st.markdown("---")

# Type distribution if available
if "type" in df_filtered.columns and df_filtered["type"].notna().any():
    st.subheader("Ticket type distribution")
    type_counts = df_filtered["type"].dropna().value_counts()
    fig_type = px.bar(
        x=type_counts.index.tolist(),
        y=type_counts.values.tolist(),
        title="Tickets per type",
        labels={"x": "Type", "y": "Tickets"},
    )
    fig_type.update_layout(
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_type, use_container_width=True)


# Raw data viewer
with st.expander("Raw filtered dataframe"):
    st.dataframe(df_filtered, use_container_width=True)
