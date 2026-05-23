import logging

from triage_agent.agent.llm import LLMFailure, decide
from triage_agent.agent.tools.topic import classify_topic
from triage_agent.agent.tools.urgency import score_urgency
from triage_agent.routing import derive_next_step
from triage_agent.schemas import AgentDecision, TriageResult

log = logging.getLogger(__name__)

SNIPPET_LENGTH = 200


def _fallback_decision(topic: str, urgency: str) -> AgentDecision:
    """Used when the LLM cannot produce a valid decision.

    Conservative: high-urgency claims escalate, very short or unclassified text
    asks for clarification, everything else forwards.
    """
    if urgency == "high":
        return AgentDecision(
            action="ESCALATE",
            reasoning="LLM unavailable; high urgency triggers conservative escalation.",
        )
    if topic == "Other":
        return AgentDecision(
            action="CLARIFY",
            reasoning="LLM unavailable; ticket did not match any topic clearly.",
            clarification_questions=[
                "Which product or service does this concern?",
                "What specifically is the issue you are facing?",
            ],
        )
    return AgentDecision(
        action="FORWARD",
        reasoning="LLM unavailable; routing by topic as fallback.",
    )


def triage(text: str, ticket_id: int) -> TriageResult:
    """Run the three-step triage flow on a single ticket."""
    topic_result = classify_topic(text)
    urgency_result = score_urgency(text)

    try:
        decision = decide(
            text=text,
            topic=topic_result.topic,
            urgency=urgency_result.level,
        )
    except LLMFailure as e:
        log.warning(
            "LLM decision failed for ticket %d, using fallback: %s", ticket_id, e
        )
        decision = _fallback_decision(topic_result.topic, urgency_result.level)

    snippet = text[:SNIPPET_LENGTH] + ("..." if len(text) > SNIPPET_LENGTH else "")

    return TriageResult(
        ticket_id=ticket_id,
        text_snippet=snippet,
        topic=topic_result.topic,
        urgency=urgency_result.level,
        action=decision.action,
        next_step=derive_next_step(decision.action, topic_result.topic),
        reasoning=decision.reasoning,
        clarification_questions=decision.clarification_questions,
    )
