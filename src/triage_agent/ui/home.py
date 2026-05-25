"""Streamlit entry point. Run with: streamlit run src/triage_agent/ui/home.py"""

import streamlit as st

st.set_page_config(
    page_title="Insurance Ticket Triage Agent",
    page_icon="\U0001f4ec",
    layout="wide",
)

st.title("Insurance Ticket Triage Agent")
st.caption(
    "Classify, prioritize and route customer support tickets "
    "using an LLM-based agentic loop."
)

st.markdown(
    """
## Pages

- **Triage Demo** - run the full pipeline end-to-end on a single ticket
  (topic embedding -> urgency classifier -> agentic decision loop -> final
  action). Three input sources are available:
  - *Insurance examples*: hand-crafted tickets covering each terminal
    action path (FORWARD, ESCALATE, CLARIFY, FAQ, CLAIM).
  - *Sample (Kaggle)*: real tickets from the prepared `sample.parquet`,
    with ground-truth labels (queue, priority, type) shown alongside the
    prediction. Includes a "random pick" button to sample any of the
    ~2000 tickets.
  - *Custom text*: paste any ticket text yourself.
- **Evaluation Dashboard** - load the triage results JSON and explore
  aggregate metrics: action distribution, confusion matrices, embedding
  margins, runtime, sample disagreements.
- **Data Statistics** - explore the prepared `sample.parquet` dataset:
  language mix, queue distribution, priority levels, ticket-length
  characteristics, and how queues map onto the insurance-context topic
  taxonomy.

## Architecture (TL;DR)

1. **Topic** is classified by a multilingual sentence-transformer
   (`BAAI/bge-m3`) via cosine similarity to five insurance-context labels:
   Policy, Claims, Billing, Technical, Other.
2. **Urgency** is scored by a hybrid: German+English keyword regex plus a
   three-class zero-shot classifier
   (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
3. **Action** is chosen by an LLM agent loop (Gemma 4 via Ollama). At each
   turn the LLM picks between helper tool (`missing_info`) and five
   terminal actions (FORWARD / ESCALATE / CLARIFY / FAQ / CLAIM). The loop
   terminates within 4 turns or falls back to a deterministic decision.

Use the left sidebar to navigate.
"""
)

st.info(
    "First run will download the embedding and zero-shot models if not cached. "
    "This can take a minute on cold start."
)
