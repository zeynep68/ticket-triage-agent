from typing import (
    Optional,
)

from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
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
