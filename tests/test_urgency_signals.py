"""Tests for the regex-only part of the urgency scorer.

The full `score_urgency` function also calls the zero-shot pipeline, which is
slow to load and requires model downloads. These tests only exercise the
keyword-signal path, which is pure logic.
"""

from triage_agent.agent.tools.urgency import _signal_score


def test_no_signals_in_neutral_text():
    score, matched = _signal_score("Please send me my invoice for May.")
    assert score == 0.0
    assert matched == []


def test_german_urgency_keyword():
    score, matched = _signal_score("Mein Auto wurde gestohlen, brauche sofort Hilfe!")
    assert score > 0.0
    assert any("gestohlen" in m for m in matched)
    assert any("sofort" in m for m in matched)


def test_english_urgency_keyword():
    score, matched = _signal_score(
        "URGENT: my account is locked out and I need access ASAP."
    )
    assert score > 0.0
    assert any("urgent" in m for m in matched)
    assert any("locked out" in m for m in matched)


def test_score_saturates_at_one():
    # Many matches still cap at 1.0
    text = (
        "urgent emergency critical asap immediately deadline stolen accident hospital"
    )
    score, _ = _signal_score(text)
    assert score == 1.0


def test_case_insensitive():
    score, matched = _signal_score("URGENT")
    assert score > 0.0
    assert "urgent" in matched
