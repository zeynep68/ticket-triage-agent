"""Agent orchestrator: runs the iterative tool-use loop on a single ticket."""

import logging

from triage_agent.agent.llm import LLMFailure, _seed_messages, step
from triage_agent.agent.prompts import (
    build_first_turn_message,
    build_followup_turn_message,
)
from triage_agent.agent.tools.missing_info import check_missing_info
from triage_agent.agent.tools.topic import classify_topic
from triage_agent.agent.tools.urgency import score_urgency
from triage_agent.routing import derive_next_step
from triage_agent.schemas import (
    HELPER_TOOLS,
    TERMINAL_TOOLS,
    AgentDecision,
    AgentStep,
    TriageResult,
)

log = logging.getLogger(__name__)

MAX_TURNS = 4
SNIPPET_LENGTH = 200


def _fallback_decision(topic: str, urgency: str) -> AgentDecision:
    """Used when the LLM loop fails or exceeds max turns.

    Conservative: high-urgency tickets escalate, unclassified text asks for
    clarification, everything else forwards.
    """
    if urgency == "high":
        return AgentDecision(
            action="ESCALATE",
            reasoning="Fallback decision (LLM unavailable): high urgency triggers conservative escalation.",
        )
    if topic == "Other":
        return AgentDecision(
            action="CLARIFY",
            reasoning="Fallback decision (LLM unavailable): ticket did not match any specific topic.",
            clarification_questions=[
                "Which product or service does this concern?",
                "What specifically is the issue you are facing?",
            ],
        )
    return AgentDecision(
        action="FORWARD",
        reasoning="Fallback decision (LLM unavailable): routing by topic.",
    )


def _terminal_to_decision(step_result: AgentStep) -> AgentDecision:
    """Convert a terminal AgentStep into an AgentDecision."""
    action_map = {
        "forward": "FORWARD",
        "escalate": "ESCALATE",
        "clarify": "CLARIFY",
        "faq": "FAQ",
    }
    action = action_map[step_result.tool]
    args = step_result.args

    return AgentDecision(
        action=action,
        reasoning=args.get("reasoning", "")[:200] or "No reasoning provided.",
        clarification_questions=args.get("clarification_questions", []),
        faq_topic=args.get("faq_topic"),
    )


def _run_loop(
    text: str,
    topic: str,
    urgency: str,
) -> tuple[AgentDecision, list[str]]:
    """Run the iterative agent loop.

    Returns (final_decision, tools_used). Falls back to a deterministic
    decision if the LLM fails or exceeds MAX_TURNS without terminating.
    """
    conversation = _seed_messages()
    conversation.append(
        {"role": "user", "content": build_first_turn_message(text, topic, urgency)}
    )

    tools_used: list[str] = []

    for turn in range(1, MAX_TURNS + 1):
        try:
            agent_step = step(conversation)
        except LLMFailure as e:
            log.warning("LLM failure at turn %d: %s", turn, e)
            return _fallback_decision(topic, urgency), tools_used + ["__fallback__"]

        tools_used.append(agent_step.tool)
        conversation.append(
            {"role": "assistant", "content": agent_step.model_dump_json()}
        )

        if agent_step.tool in TERMINAL_TOOLS:
            return _terminal_to_decision(agent_step), tools_used

        if agent_step.tool not in HELPER_TOOLS:
            log.warning(
                "Unknown tool %r at turn %d; falling back.", agent_step.tool, turn
            )
            return _fallback_decision(topic, urgency), tools_used + ["__fallback__"]

        tool_result = _dispatch_helper(agent_step, text, topic, urgency)
        conversation.append(
            {
                "role": "user",
                "content": build_followup_turn_message(
                    text=text,
                    topic=topic,
                    urgency=urgency,
                    previous_tool=agent_step.tool,
                    previous_result=tool_result,
                    turn_number=turn + 1,
                ),
            }
        )

    # Exhausted MAX_TURNS without a terminal action.
    log.warning(
        "Agent loop did not terminate within %d turns; using fallback.", MAX_TURNS
    )
    return _fallback_decision(topic, urgency), tools_used + ["__fallback__"]


def _dispatch_helper(agent_step: AgentStep, text: str, topic: str, urgency: str) -> str:
    """Execute a helper tool and return its result as a JSON-ish string for the LLM."""
    if agent_step.tool == "missing_info":
        result = check_missing_info(text=text, topic=topic, urgency=urgency)
        return result.model_dump_json()

    raise ValueError(f"Unknown helper tool: {agent_step.tool}")


def triage(text: str, ticket_id: int) -> TriageResult:
    """Run the full triage flow on a single ticket."""
    topic_result = classify_topic(text)
    urgency_result = score_urgency(text)

    decision, tools_used = _run_loop(
        text=text, topic=topic_result.topic, urgency=urgency_result.level
    )

    snippet = text[:SNIPPET_LENGTH] + ("..." if len(text) > SNIPPET_LENGTH else "")

    return TriageResult(
        ticket_id=ticket_id,
        text_snippet=snippet,
        topic=topic_result.topic,
        topic_margin=topic_result.margin,
        topic_all_scores=topic_result.all_scores,
        urgency=urgency_result.level,
        urgency_score=urgency_result.score,
        action=decision.action,
        next_step=derive_next_step(decision.action, topic_result.topic),
        reasoning=decision.reasoning,
        clarification_questions=decision.clarification_questions,
        faq_topic=decision.faq_topic,
        tools_used=tools_used,
    )
