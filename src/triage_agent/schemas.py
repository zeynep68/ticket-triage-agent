from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

TopicLiteral = Literal["Technical", "Billing", "Product", "Returns", "Outage", "Other"]
UrgencyLiteral = Literal["low", "medium", "high"]
ActionLiteral = Literal["ESCALATE", "CLARIFY", "FORWARD", "FAQ"]

# Tools the LLM can call inside the decision loop.
# Helpers gather information; terminals commit to an action.
ToolName = Literal[
    "missing_info",  # helper: check ticket completeness via LLM
    "forward",  # terminal: route to standard team
    "escalate",  # terminal: send to supervisor
    "clarify",  # terminal: ask the user for more info
    "faq",  # terminal: respond with FAQ / self-service link
]

HELPER_TOOLS: frozenset[ToolName] = frozenset({"missing_info"})
TERMINAL_TOOLS: frozenset[ToolName] = frozenset(
    {"forward", "escalate", "clarify", "faq"}
)


class CanonicalTicket(BaseModel):
    """Unified ticket schema across all CSV versions."""

    ticket_id: int
    subject: str
    body: str
    language: str
    version: str
    queue: Optional[str] = None
    priority: Optional[str] = None
    type: Optional[str] = None

    @field_validator("subject", "body")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        v = v.strip()
        if v == "":
            raise ValueError("must not be empty")
        return v.lower()[:2]

    @model_validator(mode="after")
    def has_text_content(self):
        if self.subject == "" and self.body == "":
            raise ValueError("subject and body cannot both be empty")
        return self


class TopicResult(BaseModel):
    """Output of the topic classifier."""

    topic: TopicLiteral
    margin: float
    all_scores: dict[str, float]


class UrgencyResult(BaseModel):
    """Output of the urgency scorer."""

    level: UrgencyLiteral
    score: float
    signals_found: list[str]


class MissingInfoResult(BaseModel):
    """Output of the missing-info helper tool."""

    is_actionable: bool
    missing_aspects: list[str] = []


class AgentStep(BaseModel):
    """One iteration of the agent loop.

    The LLM produces this at each turn. If `tool` is a terminal action, the
    loop ends and `args` contains the action's parameters. If `tool` is a
    helper, the loop calls it and feeds the result back into the next step.
    """

    thought: str
    tool: ToolName
    args: dict[str, Any] = {}


class AgentDecision(BaseModel):
    """LLM's terminal decision on what action to take for a ticket."""

    action: ActionLiteral
    reasoning: str
    clarification_questions: list[str] = []
    faq_topic: Optional[str] = None

    @field_validator("clarification_questions")
    @classmethod
    def cap_questions(cls, v: list[str]) -> list[str]:
        return v[:2]


class TriageResult(BaseModel):
    """Final triage output for one ticket."""

    ticket_id: int
    text_snippet: str
    topic: TopicLiteral
    topic_margin: float  # difference between top-1 and top-2 cosine similarity
    topic_all_scores: dict[str, float]  # per-topic similarity scores
    urgency: UrgencyLiteral
    urgency_score: float  # raw score in [0, 1]
    action: ActionLiteral
    next_step: str
    reasoning: str
    clarification_questions: list[str] = []
    faq_topic: Optional[str] = None
    tools_used: list[str] = []  # for traceability of the agent loop
