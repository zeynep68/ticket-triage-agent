"""Missing-info helper tool.

Uses the LLM to check whether the ticket text contains enough specific
information for a routing team to act on it. Returns a structured result the
agent can use to decide between CLARIFY and other actions.
"""

import logging

import ollama
from pydantic import ValidationError

from triage_agent.schemas import MissingInfoResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:3b-instruct"

MISSING_INFO_PROMPT = """\
You are an information-completeness checker for a customer support triage
system. Given a ticket and its pre-classified topic and urgency, decide
whether the ticket contains enough specific information for a support team
to act on it.

A ticket is actionable when at least one of the following is clear:
- which product, service, or account is concerned
- what specific problem or request the customer has
- what action the customer expects

A ticket is NOT actionable when it is too vague - for example, "help me",
"I have a problem", or generic complaints without any specific reference.

Return strict JSON in this shape:

{
  "is_actionable": true | false,
  "missing_aspects": ["product", "specific problem", ...]
}

The `missing_aspects` list should be empty when `is_actionable` is true,
and otherwise list the categories of information that are missing.
"""


def check_missing_info(
    text: str,
    topic: str,
    urgency: str,
    model: str = DEFAULT_MODEL,
) -> MissingInfoResult:
    """Run a single LLM call to assess ticket completeness."""
    user_message = f"Ticket: {text}\nPre-classified topic: {topic}\nUrgency: {urgency}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": MISSING_INFO_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 256},
    )
    raw = response["message"]["content"]

    try:
        return MissingInfoResult.model_validate_json(raw)
    except ValidationError as e:
        log.warning(
            "missing_info LLM returned invalid JSON: %s. Defaulting to actionable=True.",
            e,
        )
        # Conservative fallback: assume actionable rather than blocking with
        # a CLARIFY that may be unwarranted.
        return MissingInfoResult(is_actionable=True, missing_aspects=[])
