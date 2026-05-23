"""Orchestrator tests using mocks for the heavy dependencies.

These tests verify the orchestrator's composition logic without loading
embedding models or calling Ollama.
"""

from unittest.mock import patch

from triage_agent.agent import orchestrator
from triage_agent.agent.llm import LLMFailure
from triage_agent.schemas import AgentDecision, TopicResult, UrgencyResult


def _topic(label: str, margin: float = 0.3) -> TopicResult:
    return TopicResult(topic=label, margin=margin, all_scores={label: 0.7})


def _urgency(level: str, score: float = 0.5) -> UrgencyResult:
    return UrgencyResult(level=level, score=score, signals_found=[])


def test_forward_billing_flow():
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Billing")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "decide",
            return_value=AgentDecision(action="FORWARD", reasoning="routine billing"),
        ),
    ):
        result = orchestrator.triage("Please resend my invoice.", ticket_id=42)

    assert result.ticket_id == 42
    assert result.topic == "Billing"
    assert result.urgency == "low"
    assert result.action == "FORWARD"
    assert result.next_step == "FORWARD_BILLING"
    assert result.clarification_questions == []


def test_clarify_flow_includes_questions():
    questions = ["Which product?", "What is the issue?"]
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "decide",
            return_value=AgentDecision(
                action="CLARIFY",
                reasoning="insufficient info",
                clarification_questions=questions,
            ),
        ),
    ):
        result = orchestrator.triage("Help me", ticket_id=1)

    assert result.action == "CLARIFY"
    assert result.next_step == "ASK_CLARIFICATION"
    assert result.clarification_questions == questions


def test_escalate_flow():
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Outage")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("high", 0.9)),
        patch.object(
            orchestrator,
            "decide",
            return_value=AgentDecision(
                action="ESCALATE", reasoning="extended outage, business impact"
            ),
        ),
    ):
        result = orchestrator.triage(
            "Our office has been locked out for hours.", ticket_id=99
        )

    assert result.action == "ESCALATE"
    assert result.next_step == "ESCALATE_SUPERVISOR"


def test_llm_failure_falls_back():
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Outage")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("high")),
        patch.object(
            orchestrator, "decide", side_effect=LLMFailure("connection refused")
        ),
    ):
        result = orchestrator.triage("Some urgent outage text.", ticket_id=7)

    # Fallback escalates on high urgency
    assert result.action == "ESCALATE"
    assert result.next_step == "ESCALATE_SUPERVISOR"
    assert "fallback" in result.reasoning.lower()


def test_llm_failure_fallback_clarifies_on_other_topic():
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(orchestrator, "decide", side_effect=LLMFailure("timeout")),
    ):
        result = orchestrator.triage("Hi", ticket_id=8)

    assert result.action == "CLARIFY"
    assert result.next_step == "ASK_CLARIFICATION"
    assert len(result.clarification_questions) > 0


def test_snippet_truncates_long_text():
    long_text = "x" * 500
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "decide",
            return_value=AgentDecision(action="FORWARD", reasoning="ok"),
        ),
    ):
        result = orchestrator.triage(long_text, ticket_id=1)

    assert len(result.text_snippet) <= orchestrator.SNIPPET_LENGTH + len("...")
    assert result.text_snippet.endswith("...")
