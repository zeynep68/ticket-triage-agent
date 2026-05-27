# Insurance Ticket Triage Agent

LLM-based agent that classifies, prioritizes and routes customer support
tickets for an insurance company. Uses an iterative tool-use loop to pick
between routing actions (FORWARD / ESCALATE / CLARIFY / FAQ / CLAIM),
with a multilingual embedding model for topic classification and a
zero-shot classifier for urgency.

Built as a prototype: runs locally on a laptop.

---

## Quick start

### Local (faster, lighter, recommended for development)

Requires [uv](https://docs.astral.sh/uv/) and
[Ollama](https://ollama.com/)
installed locally. Python 3.12+ is auto-managed by uv.

```bash
uv sync --all-extras                              # install python deps + ui + dev extras
ollama serve & ollama pull qwen2.5:3b-instruct    # start ollama server in the background and pull the model
uv run prepare-data                               # write data/sample.parquet (default 2000 samples, override with --n-samples)
uv run triage --n 25                              # triage the first 25 tickets in data/sample.parquet
uv run triage-ui                                  # streamlit ui on http://localhost:8501
```

To stop the background ollama server when you are done:

```bash
pkill ollama
```

### Docker (reproducible deployment, no local python/ollama needed)

Python, uv and Ollama all run inside the containers.

```bash
docker compose up -d ollama                                # start ollama in the background, pulls qwen2.5:3b-instruct
docker compose run --rm triage-agent prepare-data          # write data/sample.parquet (default 2000 samples, override with --n-samples)
docker compose run --rm triage-agent triage --n 25         # triage run
docker compose up -d triage-agent                          # streamlit ui on http://localhost:8501
```

The container runs as root, so files written under `./data/` end up
owned by `root:root` on the host. Reclaim ownership before running
anything locally that writes to `data/` via:

```bash
sudo chown -R "$(id -u):$(id -g)" data/
```

---

## Workflows

### Data preparation

Downloads the [multilingual customer support tickets dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
from Kaggle and runs the preprocessing pipeline:

1. **Load & validate** — merge five dataset versions (v3-v5 plus two
   German normalizations). Each row is validated against a Pydantic
   schema; rejects are written to `data/rejected.csv` for audit instead
   of being silently dropped.
2. **Dedup** — identical `(subject, body)` pairs across versions are
   collapsed, keeping the row from the highest-priority version
   (v5 > v4 > v3 > ...).
3. **Filter** — keep only German and English tickets.
4. **Text cleanup** — merge `subject` + `body` into one `text` field,
   normalize whitespace, cap at 2000 chars (head + tail kept so the
   closing question survives), drop empty rows.
5. **Stratified sample** — `--n-samples` tickets (default `2000`)
   balanced across the `queue` label.

Outputs: `data/sample.parquet` and `data/rejected.csv`
(rows that failed schema validation).

```bash
# Local
uv run prepare-data --n-samples 500

# Docker
docker compose run --rm triage-agent prepare-data --n-samples 500
```

### Triage

Runs the three-stage pipeline (topic classification → urgency scorer →
agent loop) on tickets from `data/sample.parquet`. See
[Agent architecture](#agent-architecture) below for what each stage does.

`--n N` limits to the first N rows; omit to keep all tickets in
`sample.parquet`.

Output: `data/triage_results.json`.

```bash
# Local
uv run triage --n 500

# Docker
docker compose run --rm triage-agent triage --n 500
```

### Evaluation

Compares predictions in `data/triage_results.json` against the dataset's
queue and priority labels. CLI report:

```bash
# Local
uv run triage-eval

# Docker
docker compose run --rm triage-agent triage-eval
```

For an interactive view see the *Evaluation dashboard* page in the
[Streamlit UI](#streamlit-ui) below.

### Streamlit UI

Three pages:

- **Triage demo** — run the full pipeline interactively on a single
  ticket. Three input sources:
  - *Insurance examples* — hand-crafted tickets covering each terminal
    action path (FORWARD, ESCALATE, CLARIFY, FAQ, CLAIM)
  - *Sample (Kaggle)* — real tickets from `sample.parquet`, with
    ground-truth labels shown alongside the prediction
  - *Custom text* — paste any ticket text yourself
- **Evaluation dashboard** — aggregate metrics over `triage_results.json`:
  action distribution, confusion matrices, embedding margins, runtime,
  sample disagreements
- **Data statistics** — explore the prepared sample: language mix, queue
  distribution, priority levels, queue-to-topic mapping coverage

```bash
# Local
uv run triage-ui

# Docker
docker compose up -d triage-agent
open http://localhost:8501
```

### Tests

```bash
# Local
uv run pytest

# Docker
docker compose run --rm triage-agent pytest
```

---

## Agent architecture

Each ticket is processed in three stages:

1. **Topic classification** — `BAAI/bge-m3` sentence-transformer computes
   cosine similarity between the ticket text and five insurance-context
   topic descriptions (Policy / Claims / Billing / Technical / Other).
   A low-confidence fallback routes uncertain tickets to "Other".

2. **Urgency scoring** — hybrid of regex keyword matching (German +
   English urgency triggers) and binary zero-shot classification
   ("urgent and critical" vs "general inquiry") with
   `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Combined score maps to
   low / medium / high via thresholds.

3. **Agentic decision loop** — Each turn, Qwen 2.5 3B chooses between
   needing more information or committing to a terminal action:
   - **FORWARD** to the team responsible for the topic (Policy / Claims
     / Billing / Technical / Other).
   - **ESCALATE** to a human supervisor.
   - **CLARIFY** if there is not enough information to FORWARD,
     ESCALATE, FAQ, or CLAIM — ask the customer for missing details.
   - **FAQ** with a self-service link.
   - **CLAIM** — if the ticket contains enough information to create
     or update a claim record. Semantically distinct from FORWARDing to
     the Claims team: FORWARD is a routing decision if more information
     is needed to operate on that claim. In this prototype both are
     terminal markers with no downstream automation.

   Up to 4 turns; falls back to a deterministic default if the loop fails.

   In this prototype, `missing_info` only flags what is missing; the
   resulting questions are sent to the customer (via CLARIFY or attached
   to FORWARD / ESCALATE / CLAIM). In a production system, those gaps
   would first be resolved against an insurance knowledge base via RAG —
   only the questions that cannot be answered internally would reach
   the customer or the appropriate team.

See the technical writeup for design decisions, trade-offs, and
evaluation details.

---

## Models

| Component         | Model                                     | Size    | Source      |
| ----------------- | ----------------------------------------- | ------- | ----------- |
| Topic embedding   | `BAAI/bge-m3`                             | ~4.3 GB | HuggingFace |
| Urgency zero-shot | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | ~552 MB | HuggingFace |
| LLM               | `qwen2.5:3b-instruct`                     | ~1.9 GB | Ollama      |

All models are open source and run locally. No external API calls.

---

## Docker details

Two services in `docker-compose.yml`:

- **`ollama`** — pulls `ollama/ollama:latest`, auto-pulls
  `qwen2.5:3b-instruct` on first start, serves the LLM API on port 11434
  inside the docker network
- **`triage-agent`** — built from the local `Dockerfile`, runs Streamlit
  on port 8501, connects to ollama at `http://ollama:11434`

Volumes:

- `./data:/app/data` — bind mount for sample.parquet file and triage output result (JSON file)
- `hf_cache` — named volume mounted at `/root/.cache/huggingface` for
  the HuggingFace model cache
- `ollama_data` — named volume for the Qwen model
- `~/.cache/kagglehub:/root/.cache/kagglehub` — bind mount to reuse the
  host's kagglehub download cache (skipping re-download)

Teardown:

```bash
docker compose down          # stops containers, keeps volumes
docker compose down -v       # stops + removes all volumes
```

**First-time setup:** ~10-15 min total — image build installs dataset,
python packages and models.
