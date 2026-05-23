from typing import (
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
)

TopicLiteral = Literal["Policy", "Claims", "Billing", "Technical", "Other"]
UrgencyLiteral = Literal["low", "medium", "high"]
ActionLiteral = Literal["ESCALATE", "CLARIFY", "FORWARD"]


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


class AgentDecision(BaseModel):
    """LLM's decision on what action to take for a ticket."""

    action: ActionLiteral
    reasoning: str
    clarification_questions: list[str] = []

    @field_validator("clarification_questions")
    @classmethod
    def cap_questions(cls, v: list[str]) -> list[str]:
        return v[:2]


class TriageResult(BaseModel):
    """Final triage output for one ticket."""

    ticket_id: int
    text_snippet: str
    topic: TopicLiteral
    urgency: UrgencyLiteral
    action: ActionLiteral
    next_step: str
    reasoning: str
    clarification_questions: list[str] = []
