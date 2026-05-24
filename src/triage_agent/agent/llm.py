"""LLM client for the agent loop.

Each call to `step()` produces one AgentStep - either a helper tool call or
a terminal action. The orchestrator dispatches based on the step's tool.
"""

import logging

import ollama
from pydantic import ValidationError

from triage_agent.agent.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from triage_agent.schemas import AgentStep

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:3b-instruct"
MAX_PARSE_RETRIES = 2


class LLMFailure(Exception):
    """Raised when the LLM cannot produce a valid AgentStep after retries."""


def _seed_messages() -> list[dict]:
    """Build the static prefix of the conversation: system + few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    return messages


def step(
    conversation: list[dict],
    model: str = DEFAULT_MODEL,
) -> AgentStep:
    """Run one LLM turn and parse the response as an AgentStep.

    `conversation` is the full message list including system prompt, few-shot
    examples, and the running history of agent turns + tool results. The
    caller appends the new user message before calling step().
    """
    last_error: str | None = None

    for attempt in range(MAX_PARSE_RETRIES + 1):
        response = ollama.chat(
            model=model,
            messages=conversation,
            format="json",
            options={"temperature": 0.0, "num_predict": 512},
        )
        raw = response["message"]["content"]
        try:
            return AgentStep.model_validate_json(raw)
        except ValidationError as e:
            last_error = str(e)
            log.warning(
                "LLM produced invalid AgentStep on attempt %d/%d: %s",
                attempt + 1,
                MAX_PARSE_RETRIES + 1,
                last_error,
            )
            # Surface the specific Pydantic message so the model knows what to fix,
            # not just "JSON was invalid". Without this, small models tend to repeat
            # the same broken output.
            specific_errors = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            conversation.append({"role": "assistant", "content": raw})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation: {specific_errors}. "
                        "Fix the specific issue above and respond with valid JSON of "
                        'the form {"thought": "...", "tool": "...", "args": {...}}.'
                    ),
                }
            )

    raise LLMFailure(
        f"Could not obtain valid AgentStep after {MAX_PARSE_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )
