from triage_agent.schemas import ActionLiteral, TopicLiteral

FORWARD_DESTINATIONS = {
    "Billing": "FORWARD_BILLING",
    "Claims": "CREATE_CLAIM",
    "Technical": "FORWARD_TECHNICAL",
    "Policy": "FORWARD_POLICY",
    "Other": "FORWARD_GENERAL",
}


def derive_next_step(action: ActionLiteral, topic: TopicLiteral) -> str:
    """Map (action, topic) to the concrete next_step string downstream systems consume."""
    if action == "ESCALATE":
        return "ESCALATE_SUPERVISOR"
    if action == "CLARIFY":
        return "ASK_CLARIFICATION"
    return FORWARD_DESTINATIONS[topic]
