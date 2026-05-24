from functools import lru_cache
from typing import get_args

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from triage_agent.schemas import TopicLiteral, TopicResult

MODEL_NAME = "BAAI/bge-m3"

TOPIC_DESCRIPTIONS = {
    "Technical": "Individual user technical problems: login failures, password resets, account lockouts, application errors, single-user bugs, technical issues with industry-specific software.",
    "Billing": "Money and account administration: invoice questions, payment failures, billing disputes, subscription changes, credit and refund requests for charges.",
    "Product": "Specific product or feature questions: how a feature works, product defects, quality complaints, support for a particular product.",
    "Returns": "Returning purchased goods or canceling orders: return shipments, exchanges, refunds for returned items.",
    "Outage": "Service-wide problems affecting multiple users: outages, downtime, planned maintenance, infrastructure failures, system unavailability.",
    "Other": "Questions that do not fit any of the above: general inquiries, feedback, sales questions, unrelated topics.",
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
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    return labels, embeddings


def classify_topic(text: str) -> TopicResult:
    """Embed the ticket text and pick the closest topic description via cosine similarity."""
    model = _get_model()
    labels, topic_embeddings = _get_topic_embeddings()

    text_embedding = model.encode(text, convert_to_tensor=True)
    similarities = cos_sim(text_embedding, topic_embeddings)[0].tolist()

    scored = sorted(zip(labels, similarities), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = scored[0]
    second_score = scored[1][1]
    margin = top_score - second_score

    return TopicResult(
        topic=top_label,
        margin=float(margin),
        all_scores={label: float(score) for label, score in scored},
    )
