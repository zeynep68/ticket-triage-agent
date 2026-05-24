from functools import lru_cache
from typing import get_args

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from triage_agent.schemas import TopicLiteral, TopicResult

MODEL_NAME = "BAAI/bge-m3"

# Margin / score fallback: when the embedding model is genuinely uncertain
# (no topic clearly wins), route to Other instead of forcing a wrong topic.
# Both conditions must hold so that we are conservative - we only override
# when the model has neither a clear leader nor a confident match overall.
TOPIC_MARGIN_THRESHOLD = 0.01  # top-1 must beat top-2 by at least this
TOPIC_SCORE_THRESHOLD = 0.45   # top-1 must score at least this (cosine sim)

TOPIC_DESCRIPTIONS = {
    "Policy": "Insurance policy and contract matters: policy details, coverage scope, contract changes, renewals, cancellations, terms and conditions, tariff changes, plan upgrades or downgrades.",
    "Claims": "Damage claims, incidents, and benefit requests: reporting damage, theft, accidents, medical incidents, requesting reimbursement for losses, claim status updates, hardware or service failures the customer wants compensated.",
    "Billing": "Payments, invoices, and account finance: invoice questions, payment failures, direct debit issues, premium adjustments, refunds for overcharges, dunning, subscription billing disputes.",
    "Technical": "Online portal, app, and account access: login failures, password resets, app errors, portal unavailability, account lockouts, technical issues with self-service tools.",
    "Other": "Off-topic or unrelated to insurance support: general inquiries, sales questions, feedback, requests outside our service scope, vague messages, or topics like fitness, hobbies, books, travel, or unrelated industries.",
}

assert set(TOPIC_DESCRIPTIONS.keys()) == set(get_args(TopicLiteral)), (
    f"TOPIC_DESCRIPTIONS must cover all TopicLiteral values. "
    f"Missing: {set(get_args(TopicLiteral)) - set(TOPIC_DESCRIPTIONS.keys())}. "
    f"Extra: {set(TOPIC_DESCRIPTIONS.keys()) - set(get_args(TopicLiteral))}."
)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_topic_embeddings():
    model = _get_model()
    labels = list(TOPIC_DESCRIPTIONS.keys())
    descriptions = list(TOPIC_DESCRIPTIONS.values())
    embeddings = model.encode(
        descriptions, convert_to_tensor=True, show_progress_bar=False
    )
    return labels, embeddings


def classify_topic(text: str) -> TopicResult:
    """Embed the ticket text and pick the closest topic description via cosine similarity.

    If the model is genuinely uncertain (low margin AND low top score), fall back
    to "Other" rather than forcing a misleading specific label. This makes the
    classifier honest about its uncertainty.
    """
    model = _get_model()
    labels, topic_embeddings = _get_topic_embeddings()

    text_embedding = model.encode(
        text, convert_to_tensor=True, show_progress_bar=False
    )
    similarities = cos_sim(text_embedding, topic_embeddings)[0].tolist()

    scored = sorted(zip(labels, similarities), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = scored[0]
    second_score = scored[1][1]
    margin = top_score - second_score
    all_scores = {label: float(score) for label, score in scored}

    # Uncertainty fallback to "Other": both conditions must hold (conservative).
    if margin < TOPIC_MARGIN_THRESHOLD and top_score < TOPIC_SCORE_THRESHOLD:
        return TopicResult(topic="Other", margin=float(margin), all_scores=all_scores)

    return TopicResult(
        topic=top_label,
        margin=float(margin),
        all_scores=all_scores,
    )
