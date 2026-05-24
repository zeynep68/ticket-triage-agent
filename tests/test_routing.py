import pytest

from triage_agent.routing import derive_next_step


def test_escalate_overrides_topic():
    assert derive_next_step("ESCALATE", "Billing") == "ESCALATE_SUPERVISOR"
    assert derive_next_step("ESCALATE", "Technical") == "ESCALATE_SUPERVISOR"


def test_clarify_overrides_topic():
    assert derive_next_step("CLARIFY", "Technical") == "ASK_CLARIFICATION"
    assert derive_next_step("CLARIFY", "Other") == "ASK_CLARIFICATION"


def test_faq_overrides_topic():
    assert derive_next_step("FAQ", "Technical") == "SEND_FAQ_LINK"
    assert derive_next_step("FAQ", "Billing") == "SEND_FAQ_LINK"


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("Technical", "FORWARD_TECHNICAL"),
        ("Billing", "FORWARD_BILLING"),
        ("Product", "FORWARD_PRODUCT"),
        ("Returns", "FORWARD_RETURNS"),
        ("Outage", "FORWARD_OUTAGE"),
        ("Other", "FORWARD_GENERAL"),
    ],
)
def test_forward_routes_by_topic(topic, expected):
    assert derive_next_step("FORWARD", topic) == expected
