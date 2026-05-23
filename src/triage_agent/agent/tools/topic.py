from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from triage_agent.schemas import TopicResult

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TOPIC_DESCRIPTIONS = {
    "Policy": "Insurance policy and contract questions: coverage terms, cancellation, renewal, policy changes.",
    "Claims": "Claims and damage: filing a claim, damage assessment, payout status, accident or incident report.",
    "Billing": "Billing and payment: invoice issues, payment problems, refund requests, premium charges.",
    "Technical": "Technical and online access: login problems, app errors, account lockout, password reset.",
    "Other": "General inquiries that do not fit Policy, Claims, Billing, or Technical categories.",
}


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
