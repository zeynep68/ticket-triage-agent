from typing import get_args

from triage_agent.schemas import ActionLiteral, TopicLiteral

FORWARD_DESTINATIONS = {
    "Policy": "FORWARD_POLICY",
    "Claims": "FORWARD_CLAIMS",
    "Billing": "FORWARD_BILLING",
    "Technical": "FORWARD_TECHNICAL",
    "Other": "FORWARD_GENERAL",
}

assert set(FORWARD_DESTINATIONS.keys()) == set(get_args(TopicLiteral)), (
    f"FORWARD_DESTINATIONS must cover all TopicLiteral values. "
    f"Missing: {set(get_args(TopicLiteral)) - set(FORWARD_DESTINATIONS.keys())}. "
    f"Extra: {set(FORWARD_DESTINATIONS.keys()) - set(get_args(TopicLiteral))}."
)


def derive_next_step(action: ActionLiteral, topic: TopicLiteral) -> str:
    """Map (action, topic) to the concrete next_step string downstream systems consume."""
    if action == "ESCALATE":
        return "ESCALATE_SUPERVISOR"
    if action == "CLARIFY":
        return "ASK_CLARIFICATION"
    if action == "FAQ":
        return "SEND_FAQ_LINK"
    if action == "CLAIM":
        return "CREATE_OR_UPDATE_CLAIM"
    return FORWARD_DESTINATIONS[topic]
