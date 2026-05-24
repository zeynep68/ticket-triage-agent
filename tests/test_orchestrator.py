"""Orchestrator tests using mocks for the heavy dependencies.

These tests verify the agent loop's composition and dispatch logic without
loading embedding models or calling Ollama.
"""

from unittest.mock import patch

from triage_agent.agent import orchestrator
from triage_agent.agent.llm import LLMFailure
from triage_agent.schemas import (
    AgentStep,
    MissingInfoResult,
    TopicResult,
    UrgencyResult,
)


def _topic(label: str, margin: float = 0.3) -> TopicResult:
    return TopicResult(topic=label, margin=margin, all_scores={label: 0.7})


def _urgency(level: str, score: float = 0.5) -> UrgencyResult:
    return UrgencyResult(level=level, score=score, signals_found=[])


def _terminal_step(tool: str, **args) -> AgentStep:
    return AgentStep(thought="test", tool=tool, args=args)


def test_direct_forward_path():
    """Single-turn: LLM commits to forward immediately."""
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Billing")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "step",
            return_value=_terminal_step("forward", reasoning="routine billing"),
        ),
    ):
        result = orchestrator.triage("Please resend my invoice.", ticket_id=42)

    assert result.ticket_id == 42
    assert result.topic == "Billing"
    assert result.urgency == "low"
    assert result.action == "FORWARD"
    assert result.next_step == "FORWARD_BILLING"
    assert result.tools_used == ["forward"]


def test_direct_faq_path():
    """Single-turn: LLM commits to FAQ."""
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Technical")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "step",
            return_value=_terminal_step(
                "faq",
                reasoning="how-to question",
                faq_topic="password_reset",
            ),
        ),
    ):
        result = orchestrator.triage("How do I reset my password?", ticket_id=1)

    assert result.action == "FAQ"
    assert result.next_step == "SEND_FAQ_LINK"
    assert result.faq_topic == "password_reset"


def test_missing_info_then_clarify_path():
    """Two-turn: LLM calls missing_info first, then clarifies."""
    steps = [
        AgentStep(thought="vague, check info", tool="missing_info", args={}),
        _terminal_step(
            "clarify",
            reasoning="info confirmed missing",
            clarification_questions=["Which product?", "What problem?"],
        ),
    ]

    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(orchestrator, "step", side_effect=steps),
        patch.object(
            orchestrator,
            "check_missing_info",
            return_value=MissingInfoResult(
                is_actionable=False, missing_aspects=["product", "problem"]
            ),
        ),
    ):
        result = orchestrator.triage("Help me", ticket_id=1)

    assert result.action == "CLARIFY"
    assert result.next_step == "ASK_CLARIFICATION"
    assert result.clarification_questions == ["Which product?", "What problem?"]
    assert result.tools_used == ["missing_info", "clarify"]


def test_llm_failure_falls_back():
    """LLM error in the loop triggers the deterministic fallback."""
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Technical")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("high")),
        patch.object(
            orchestrator, "step", side_effect=LLMFailure("connection refused")
        ),
    ):
        result = orchestrator.triage("Some urgent outage text.", ticket_id=7)

    assert result.action == "ESCALATE"  # fallback escalates on high urgency
    assert result.next_step == "ESCALATE_SUPERVISOR"
    assert "fallback" in result.reasoning.lower()
    assert "__fallback__" in result.tools_used


def test_repeated_missing_info_breaks_loop():
    """If LLM calls missing_info twice in a row, hard guard breaks the loop."""
    # Always returns missing_info; the guard should fire on turn 2.
    looping_step = AgentStep(thought="loop", tool="missing_info", args={})

    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(orchestrator, "step", return_value=looping_step),
        patch.object(
            orchestrator,
            "check_missing_info",
            return_value=MissingInfoResult(is_actionable=True),
        ),
    ):
        result = orchestrator.triage("Some text.", ticket_id=8)

    assert "fallback" in result.reasoning.lower()
    assert "__loop_break__" in result.tools_used
    # Should have called missing_info exactly once before the guard fired
    assert result.tools_used.count("missing_info") == 1


def test_snippet_truncates_long_text():
    long_text = "x" * 500
    with (
        patch.object(orchestrator, "classify_topic", return_value=_topic("Other")),
        patch.object(orchestrator, "score_urgency", return_value=_urgency("low")),
        patch.object(
            orchestrator,
            "step",
            return_value=_terminal_step("forward", reasoning="ok"),
        ),
    ):
        result = orchestrator.triage(long_text, ticket_id=1)

    assert len(result.text_snippet) <= orchestrator.SNIPPET_LENGTH + len("...")
    assert result.text_snippet.endswith("...")
