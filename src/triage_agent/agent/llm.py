import logging

import ollama
from pydantic import ValidationError

from triage_agent.agent.prompts import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_user_message,
)
from triage_agent.schemas import AgentDecision

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:3b-instruct"
MAX_RETRIES = 2


class LLMFailure(Exception):
    """Raised when the LLM cannot produce a valid AgentDecision after retries."""


def _build_messages(text: str, topic: str, urgency: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    messages.append({"role": "user", "content": build_user_message(text, topic, urgency)})
    return messages


def decide(
    text: str,
    topic: str,
    urgency: str,
    model: str = DEFAULT_MODEL,
) -> AgentDecision:
    """Ask the LLM for a triage decision. Retries on JSON-parse failure."""
    messages = _build_messages(text, topic, urgency)

    last_error: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        response = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"]
        try:
            return AgentDecision.model_validate_json(raw)
        except ValidationError as e:
            last_error = str(e)
            log.warning(
                "LLM produced invalid JSON on attempt %d/%d: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                last_error,
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON or did not match "
                        "the required schema. Please respond with valid JSON only, "
                        "matching the action/reasoning/clarification_questions shape."
                    ),
                }
            )

    raise LLMFailure(
        f"Could not obtain valid AgentDecision after {MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )
