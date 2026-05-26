# syntax=docker/dockerfile:1.7
# Python + uv base. Slim variant keeps the image small while still having apt.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Layer 1: dependency manifests only. Changes rarely; this layer is reused
# whenever only source code changes, which keeps rebuilds fast.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --all-extras

# Layer 2: source code. This invalidates only when src/ changes.
COPY src/ ./src/
RUN uv sync --frozen --all-extras

# Connectivity to the Ollama sibling container (resolved by docker-compose DNS).
ENV OLLAMA_HOST=http://ollama:11434

# Persist the HuggingFace cache to a named volume mounted at this path.
ENV HF_HOME=/root/.cache/huggingface

# Stream logs to stdout immediately instead of buffering.
ENV PYTHONUNBUFFERED=1

# Streamlit default port. docker-compose maps it to the host.
EXPOSE 8501

# Default command starts the Streamlit UI. CLI tasks (prepare-data, triage,
# pytest, run_eval) are run via `docker compose run --rm triage-agent <cmd>`
# which overrides this default.
CMD ["uv", "run", "streamlit", "run", "src/triage_agent/ui/home.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
