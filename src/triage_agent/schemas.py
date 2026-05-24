from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

TopicLiteral = Literal["Policy", "Claims", "Billing", "Technical", "Other"]
UrgencyLiteral = Literal["low", "medium", "high"]
ActionLiteral = Literal["ESCALATE", "CLARIFY", "FORWARD", "FAQ", "CLAIM"]

# Tools the LLM can call inside the decision loop.
# Helpers gather information; terminals commit to an action.
ToolName = Literal[
    "missing_info",  # helper: check ticket completeness via LLM
    "forward",  # terminal: route to standard team
    "escalate",  # terminal: send to supervisor
    "clarify",  # terminal: ask the user for more info
    "faq",  # terminal: respond with FAQ / self-service link
    "claim",  # terminal: create or update an insurance claim
]

HELPER_TOOLS: frozenset[ToolName] = frozenset({"missing_info"})
TERMINAL_TOOLS: frozenset[ToolName] = frozenset(
    {"forward", "escalate", "clarify", "faq", "claim"}
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

    @model_validator(mode="after")
    def clarify_step_requires_questions(self):
        # Catches the failure case at the LLM-output boundary so step()'s
        # retry loop self-corrects via the existing ValidationError handler.
        if self.tool == "clarify":
            questions = self.args.get("clarification_questions", [])
            if not isinstance(questions, list):
                raise ValueError("clarification_questions must be a list")
            valid = [q for q in questions if isinstance(q, str) and q.strip()]
            if not valid:
                raise ValueError(
                    "clarify tool requires at least one non-empty "
                    "clarification_question in args; the customer needs "
                    "something to answer."
                )
        return self


class AgentDecision(BaseModel):
    """LLM's terminal decision on what action to take for a ticket."""

    action: ActionLiteral
    reasoning: str
    clarification_questions: list[str] = []
    faq_topic: Optional[str] = None

    @field_validator("clarification_questions")
    @classmethod
    def cap_questions(cls, v: list[str]) -> list[str]:
        # Drop empty/whitespace-only entries, then cap at 2.
        return [q for q in v if q and q.strip()][:2]

    @model_validator(mode="after")
    def clarify_requires_questions(self):
        # CLARIFY is semantically meaningless without questions to send back.
        # Prompt instructs the LLM to provide them, but small/quantized models
        # sometimes drop the field - this is the safety net that triggers a
        # retry in step() via ValidationError.
        if self.action == "CLARIFY" and not self.clarification_questions:
            raise ValueError(
                "CLARIFY requires at least one clarification_question; "
                "the customer needs to know what to answer."
            )
        return self


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
