"""Live triage page: paste a ticket, watch the pipeline run, see the decision."""

import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Heavy ML-library imports (orchestrator pulls in torch + sentence-transformers
# + transformers + ollama) are deferred to the run_button branch so the page
# itself loads instantly. First triage triggers the imports under the spinner.
from triage_agent.ui.components.triage_rendering import (
    render_decision_section,
    render_loop_trace,
    render_runtime,
    render_topic_section,
    render_urgency_section,
)
from triage_agent.ui.example_tickets import EXAMPLE_TICKETS

st.set_page_config(
    page_title="Triage Demo",
    page_icon="\U0001f4ec",
    layout="wide",
)

st.title("Triage Demo")
st.caption(
    "Paste a ticket text, pick a curated insurance example, or load a real "
    "ticket from the Kaggle sample. The full pipeline runs end-to-end."
)

DEFAULT_SAMPLE_PATH = Path("data/sample.parquet")
TEXTAREA_KEY = "ticket_text_input"


@st.cache_data(show_spinner=False)
def _load_sample_tickets(path_str: str, mtime: float) -> pd.DataFrame:
    """Cached by (path, mtime) so we only re-read when the parquet changes."""
    return pd.read_parquet(path_str)


def _format_sample_label(row: pd.Series) -> str:
    subject = (row.get("subject") or "").strip()
    truncated = subject[:60] + ("..." if len(subject) > 60 else "")
    return f"{row['ticket_id']} - {truncated}"


def _set_textarea(value: str, identity: str) -> None:
    """Force-update the textarea content and remember which selection it came from."""
    st.session_state[TEXTAREA_KEY] = value
    st.session_state["_text_identity"] = identity


with st.sidebar:
    st.header("Input")

    source = st.radio(
        "Source",
        options=("Insurance examples", "Sample (Kaggle)", "Custom text"),
        index=0,
        help=(
            "Insurance examples are hand-written tickets covering each action "
            "path. Sample loads real tickets from data/sample.parquet."
        ),
    )

    selected_row: pd.Series | None = None
    identity = ""
    default_text = ""

    if source == "Insurance examples":
        example_label = st.selectbox(
            "Curated example",
            options=list(EXAMPLE_TICKETS.keys()),
            index=0,
        )
        identity = f"insurance:{example_label}"
        default_text = EXAMPLE_TICKETS[example_label]

    elif source == "Sample (Kaggle)":
        sample_path = st.text_input(
            "Sample parquet path",
            value=str(DEFAULT_SAMPLE_PATH),
        )
        path = Path(sample_path)
        if not path.exists():
            st.error(f"File not found: {path}")
            st.stop()

        df = _load_sample_tickets(str(path), path.stat().st_mtime)
        st.caption(f"{len(df):,} tickets available in this sample")

        df_sorted = df.sort_values("ticket_id").head(50).reset_index(drop=True)
        labels = [_format_sample_label(row) for _, row in df_sorted.iterrows()]

        # Track whether the current sample selection comes from the dropdown
        # or from a "random" click. The random pick must survive the rerun
        # without being clobbered by the dropdown's default index.
        idx = st.selectbox(
            "Ticket",
            options=range(len(df_sorted)),
            format_func=lambda i: labels[i],
        )
        prev_idx = st.session_state.get("_last_dropdown_idx")
        dropdown_changed = prev_idx is not None and idx != prev_idx
        st.session_state["_last_dropdown_idx"] = idx

        if st.button("Pick random from full sample", use_container_width=True):
            random_row = df.sample(n=1, random_state=random.randint(0, 10**6)).iloc[0]
            st.session_state["_random_row"] = random_row.to_dict()
            st.session_state["_sample_mode"] = "random"
            _set_textarea(
                random_row["text"], f"sample:random:{random_row['ticket_id']}"
            )
            st.rerun()

        # Mode resolution: dropdown change wins (user explicitly picked one);
        # otherwise keep the previous mode (which may be "random" from an
        # earlier button click that survived a rerun).
        if dropdown_changed:
            st.session_state["_sample_mode"] = "dropdown"
        mode = st.session_state.get("_sample_mode", "dropdown")

        if mode == "random" and "_random_row" in st.session_state:
            selected_row = pd.Series(st.session_state["_random_row"])
            identity = f"sample:random:{selected_row['ticket_id']}"
        else:
            selected_row = df_sorted.iloc[idx]
            identity = f"sample:{selected_row['ticket_id']}"
        default_text = selected_row["text"]

    else:  # Custom text
        identity = "custom"
        default_text = ""

    # Reset the textarea whenever the selection identity changes.
    # This is what was broken before: without this, the textarea stuck on the
    # previous source's text because session_state held its value across reruns.
    if st.session_state.get("_text_identity") != identity:
        _set_textarea(default_text, identity)

    ticket_text = st.text_area(
        "Ticket text (subject + body)",
        height=220,
        placeholder="Z.B. 'Mein Auto wurde gestohlen, Vertragsnummer KFZ-12345'",
        key=TEXTAREA_KEY,
    )

    run_button = st.button("Triage starten", type="primary", use_container_width=True)

    if source == "Sample (Kaggle)" and selected_row is not None and len(selected_row) > 0:
        st.markdown("---")
        st.markdown("**Ground truth (Kaggle labels)**")
        gt_lines = []
        for field in ("queue", "priority", "type", "language"):
            value = selected_row.get(field)
            if value is not None and str(value) not in ("nan", ""):
                gt_lines.append(f"- `{field}`: {value}")
        st.markdown("\n".join(gt_lines) if gt_lines else "_(no labels)_")

    st.markdown("---")
    st.caption(
        "First run loads the embedding + zero-shot models (~3 GB total). "
        "Subsequent runs use cached models."
    )


if run_button and ticket_text.strip():
    # Deferred imports: keep them inside the run path so the page loads instantly.
    with st.spinner("Loading models (first run only)..."):
        from triage_agent.agent.orchestrator import triage
        from triage_agent.agent.tools.topic import classify_topic
        from triage_agent.agent.tools.urgency import score_urgency

    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.spinner("Classifying topic..."):
            topic_result = classify_topic(ticket_text)
        render_topic_section(topic_result)

    with col_right:
        with st.spinner("Scoring urgency..."):
            urgency_result = score_urgency(ticket_text)
        render_urgency_section(urgency_result)

    st.markdown("---")

    with st.spinner("Running agent loop..."):
        start = time.perf_counter()
        result = triage(text=ticket_text, ticket_id=0)
        elapsed = time.perf_counter() - start

    col_trace, col_decision = st.columns([1, 1])
    with col_trace:
        render_loop_trace(result.tools_used)
    with col_decision:
        render_decision_section(result)

    render_runtime(elapsed)

    with st.expander("Raw TriageResult"):
        raw = result.model_dump()
        # If we ran on a sample-parquet ticket, include the Kaggle ground-truth
        # labels so they're visible alongside the prediction in the same JSON.
        if selected_row is not None and len(selected_row) > 0:
            raw["ground_truth"] = {
                field: (
                    None
                    if pd.isna(selected_row.get(field))
                    else selected_row.get(field)
                )
                for field in ("queue", "priority", "type", "language")
            }
        st.json(raw)

elif run_button:
    st.warning("Bitte gib einen Ticket-Text ein oder wähle ein Beispiel.")
else:
    st.info(
        "Wähle eine Quelle in der Sidebar (Insurance Beispiele oder ein "
        "Sample-Ticket aus dem Kaggle-Datensatz), dann klicke **Triage starten**."
    )
