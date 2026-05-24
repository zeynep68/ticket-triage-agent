"""Maps the dataset's `queue` and `priority` fields onto our taxonomies.

These mappings exist only at evaluation time. The agent never sees them - they
are used to compare predictions against the dataset's labels.

The dataset uses ~30 queue categories spanning many domains (IT, billing, arts,
home & garden, etc.). Our taxonomy is 5 insurance-context topics: Policy, Claims,
Billing, Technical, Other. The mapping is best-effort: the Kaggle dataset is
generic IT/customer support, not actually insurance, so many queues do not have
a clean insurance analog and fall through to Other.
"""

import pandas as pd

from triage_agent.schemas import TopicLiteral, UrgencyLiteral

QUEUE_TO_TOPIC: dict[str, TopicLiteral] = {
    # Technical: online portal, app, account access, system tooling
    "Technical Support": "Technical",
    "IT Support": "Technical",
    "IT & Technology/Security Operations": "Technical",
    "IT & Technology/Network Infrastructure": "Technical",
    "IT & Technology/Software Development": "Technical",
    "Network Infrastructure": "Technical",
    "Security Operations": "Technical",
    "Software Development": "Technical",
    "Product Support": "Technical",
    # Billing: invoices, payments, premiums, account finance
    "Billing and Payments": "Billing",
    "Account & Billing Management": "Billing",
    # Claims: damage, hardware failure, returns/refunds, service outages
    # (the customer is reporting something broken or seeking compensation)
    "Returns and Exchanges": "Claims",
    "Service Outages and Maintenance": "Claims",
    "IT & Technology/Hardware Support": "Claims",
    "Health/Medical Services": "Claims",
    # Policy: pre-sales, contract changes, release planning
    "Sales and Pre-Sales": "Policy",
    "Change Management": "Policy",
    "Release Management": "Policy",
    # Everything else (Customer Service, General Inquiry, vertical-specific
    # queues like Pets/Movies/Home & Garden, etc.) falls through to "Other"
    # via the default in map_queue_to_topic.
}


def map_queue_to_topic(queue: str | None) -> TopicLiteral | None:
    """Map a dataset queue value to our 6-topic taxonomy.

    Returns None if the queue itself is None (missing data). Returns "Other"
    for queues that exist but do not map to any specific topic - that matches
    how the agent classifies non-matching tickets.
    """
    if queue is None or (isinstance(queue, float) and pd.isna(queue)):
        return None
    return QUEUE_TO_TOPIC.get(queue, "Other")


def map_priority_to_urgency(priority: str | None) -> UrgencyLiteral | None:
    """Map dataset priority to our urgency literal. Same vocabulary, just normalize case."""
    if priority is None or (isinstance(priority, float) and pd.isna(priority)):
        return None
    p = priority.strip().lower()
    if p in ("low", "medium", "high"):
        return p  # type: ignore[return-value]
    if p in ("critical", "urgent"):
        return "high"
    if p in ("normal", "standard"):
        return "medium"
    if p == "very_low":
        return "low"
    return None
