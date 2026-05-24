"""Tests for the evaluation report's data-building logic.

The report-printing functions (report_overall, report_urgency, etc.) are
side-effect-only and rely on visual inspection. We test the data structure
that feeds them.
"""

from triage_agent.evaluation.run_eval import URGENCY_NUMERIC, build_dataframe


def _sample_result(
    topic: str = "Billing",
    urgency: str = "low",
    action: str = "FORWARD",
    queue: str | None = "Billing and Payments",
    priority: str | None = "low",
    tools_used: list[str] | None = None,
) -> dict:
    return {
        "ticket_id": 1,
        "topic": topic,
        "urgency": urgency,
        "action": action,
        "next_step": "FORWARD_BILLING",
        "reasoning": "x",
        "text_snippet": "a snippet",
        "clarification_questions": [],
        "faq_topic": None,
        "tools_used": tools_used or ["forward"],
        "ground_truth": {"queue": queue, "priority": priority, "type": None},
    }


def test_build_dataframe_maps_ground_truth_correctly():
    df = build_dataframe([_sample_result()])
    row = df.iloc[0]
    assert row["mapped_topic"] == "Billing"
    assert row["mapped_urgency"] == "low"


def test_unmapped_queue_falls_through_to_other():
    df = build_dataframe([_sample_result(queue="Arts & Entertainment/Movies")])
    assert df.iloc[0]["mapped_topic"] == "Other"


def test_missing_priority_results_in_none():
    df = build_dataframe([_sample_result(priority=None)])
    assert df.iloc[0]["mapped_urgency"] is None


def test_missing_queue_results_in_none():
    df = build_dataframe([_sample_result(queue=None)])
    assert df.iloc[0]["mapped_topic"] is None


def test_tools_used_is_tuple_for_grouping():
    df = build_dataframe([_sample_result(tools_used=["missing_info", "clarify"])])
    # Tuples are hashable and groupable; lists are not.
    assert isinstance(df.iloc[0]["tools_used"], tuple)
    assert df.iloc[0]["tools_used"] == ("missing_info", "clarify")


def test_urgency_numeric_mapping_is_consistent():
    # Off-by-one logic relies on this mapping being ordered correctly.
    assert URGENCY_NUMERIC["low"] < URGENCY_NUMERIC["medium"]
    assert URGENCY_NUMERIC["medium"] < URGENCY_NUMERIC["high"]


def test_urgency_offbyone_computation():
    """Verify the |pred - gt| <= 1 logic works on the dataframe."""
    results = [
        _sample_result(urgency="low", priority="low"),  # exact match
        _sample_result(urgency="medium", priority="low"),  # off by 1
        _sample_result(urgency="high", priority="low"),  # off by 2
    ]
    df = build_dataframe(results).copy()
    df["pred_num"] = df["predicted_urgency"].map(URGENCY_NUMERIC)
    df["gt_num"] = df["mapped_urgency"].map(URGENCY_NUMERIC)

    exact = (df["predicted_urgency"] == df["mapped_urgency"]).sum()
    one_off = (df["pred_num"] - df["gt_num"]).abs().le(1).sum()

    assert exact == 1  # only the first one
    assert one_off == 2  # first (0 diff) + second (1 diff), not third (2 diff)


def test_action_distribution_via_value_counts():
    results = [
        _sample_result(action="FORWARD"),
        _sample_result(action="FORWARD"),
        _sample_result(action="ESCALATE"),
        _sample_result(action="CLARIFY"),
    ]
    df = build_dataframe(results)
    counts = df["action"].value_counts().to_dict()
    assert counts == {"FORWARD": 2, "ESCALATE": 1, "CLARIFY": 1}


def test_handles_empty_results():
    df = build_dataframe([])
    assert len(df) == 0


def test_handles_priority_case_normalization():
    """Critical priority gets mapped to 'high'."""
    df = build_dataframe([_sample_result(priority="critical")])
    assert df.iloc[0]["mapped_urgency"] == "high"


def test_handles_very_low_priority():
    df = build_dataframe([_sample_result(priority="very_low")])
    assert df.iloc[0]["mapped_urgency"] == "low"


def test_fallback_marker_in_tools_used():
    df = build_dataframe([_sample_result(tools_used=["missing_info", "__fallback__"])])
    assert "__fallback__" in df.iloc[0]["tools_used"]
