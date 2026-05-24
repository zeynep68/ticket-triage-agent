"""Tests for the dataset queue/priority -> our taxonomy mapping."""

import pytest

from triage_agent.evaluation.ground_truth import (
    map_priority_to_urgency,
    map_queue_to_topic,
)


@pytest.mark.parametrize(
    "queue,expected",
    [
        ("Technical Support", "Technical"),
        ("IT Support", "Technical"),
        ("Product Support", "Technical"),
        ("Billing and Payments", "Billing"),
        ("Account & Billing Management", "Billing"),
        ("Returns and Exchanges", "Claims"),
        ("Service Outages and Maintenance", "Claims"),
        ("IT & Technology/Hardware Support", "Claims"),
        ("Health/Medical Services", "Claims"),
        ("Sales and Pre-Sales", "Policy"),
        ("Change Management", "Policy"),
        ("Release Management", "Policy"),
    ],
)
def test_known_queues_map_correctly(queue, expected):
    assert map_queue_to_topic(queue) == expected


def test_customer_service_defaults_to_other():
    # Customer Service has no explicit mapping; should fall through to default.
    assert map_queue_to_topic("Customer Service") == "Other"


def test_unknown_queue_maps_to_other():
    assert map_queue_to_topic("Arts & Entertainment/Movies") == "Other"
    assert map_queue_to_topic("Home & Garden/Landscaping") == "Other"


def test_none_queue_returns_none():
    assert map_queue_to_topic(None) is None


@pytest.mark.parametrize(
    "priority,expected",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("LOW", "low"),
        ("High", "high"),
        ("critical", "high"),
        ("urgent", "high"),
        ("normal", "medium"),
        ("very_low", "low"),
    ],
)
def test_priority_mapping(priority, expected):
    assert map_priority_to_urgency(priority) == expected


def test_unknown_priority_returns_none():
    assert map_priority_to_urgency("weird-value") is None
    assert map_priority_to_urgency(None) is None
