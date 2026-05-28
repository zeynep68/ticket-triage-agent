"""Integration tests that run the full pipeline with real models.

Requires:
  - Ollama running with qwen2.5:3b-instruct loaded
  - HuggingFace models downloaded (BGE-M3, mDeBERTa)
"""

from triage_agent.agent.orchestrator import triage


def test_out_of_scope_ticket():
    """A ticket unrelated to insurance should not be routed to a real team."""
    result = triage(
        "I have a question about dinosaurs. When did they live?", ticket_id=9
    )

    assert result.topic == "Other"


def test_faq_password_reset():
    """A generic how-to question should route to FAQ."""
    result = triage(
        "I need to reset my password. Can you explain me how to do that?", ticket_id=1
    )

    assert result.topic == "Technical"
    assert result.urgency == "low"
    assert result.action == "FAQ"
    assert result.next_step == "SEND_FAQ_LINK"


def test_forward_billing():
    """A routine billing ticket should forward to billing."""
    result = triage(
        "I haven't received my invoice for last month. "
        "Can you please resend it to my email? My mail is example@email.com",
        ticket_id=2,
    )

    assert result.topic == "Billing"
    assert result.urgency == "low"
    assert result.action == "FORWARD"
    assert result.next_step == "FORWARD_BILLING"


def test_claim_theft():
    """A theft report with urgency signals should create a claim."""
    result = triage(
        "Mein Auto wurde gestern Nacht gestohlen. "
        "Ich habe bereits eine Anzeige bei der Polizei erstattet. "
        "Ich brauche dringend Unterstützung bei der Schadensmeldung. "
        "Vertragsnummer: VP-2024-88123.",
        ticket_id=3,
    )

    assert result.topic == "Claims"
    assert result.urgency == "high"
    assert result.action == "CLAIM"
    assert result.next_step == "CREATE_OR_UPDATE_CLAIM"


def test_escalate_lawyer_threat():
    """Mention of a lawyer should trigger escalation."""
    result = triage(
        "I've contacted my lawyer about this. My claim #CL-9912 "
        "has been ignored for three months despite multiple follow-ups. "
        "I expect a resolution immediately.",
        ticket_id=4,
    )

    assert result.action == "ESCALATE"
    assert result.next_step == "ESCALATE_SUPERVISOR"


def test_clarify_vague_ticket():
    """A vague ticket with no identifiable topic should ask for clarification."""
    result = triage("Help me", ticket_id=5)

    assert result.action == "CLARIFY"
    assert result.next_step == "ASK_CLARIFICATION"
    assert len(result.clarification_questions) > 0


def test_forward_policy():
    """A policy question should forward to the policy team."""
    result = triage(
        "I would like to change the coverage on my home insurance policy. ",
        ticket_id=6,
    )

    assert result.topic == "Policy"
    assert result.action == "FORWARD"
    assert result.next_step == "FORWARD_POLICY"


def test_high_urgency_water_damage():
    """Active water damage should be high urgency."""
    result = triage(
        "Wasserrohrbruch in meiner Wohnung, alles steht unter Wasser. "
        "Brauche sofort Hilfe! Bitte melden Sie sich so schnell wie möglich. ",
        ticket_id=7,
    )

    assert result.urgency == "high"
    assert result.topic == "Claims"


def test_very_short_ticket():
    """A very short ticket with no identifiable topic should ask for clarification."""
    result = triage("Help me", ticket_id=5)

    assert result.action == "CLARIFY"
    assert result.next_step == "ASK_CLARIFICATION"
    assert len(result.clarification_questions) > 0


def test_complex_multi_topic_ticket():
    """A ticket touching two topics should still route to one team."""
    result = triage(
        "Ich habe eine Frage zu meinem Auslandsversicherungsvertrag. "
        "Für meine nächste Reise nach Spanien möchte ich wissen, ob ich auch dort einen Schaden melden kann, falls etwas passiert. "
        "Außerdem wollte ich mich bezüglich einer KFZ Versicherung informieren. Können Sie mir sagen, welche Konditionen für die Autoversicherung gelten? ",
        ticket_id=8,
    )

    assert result.topic in ("Policy", "Clarify")
    assert result.action in ("FORWARD", "FAQ", "CLARIFY")
    assert result.urgency == "low"
