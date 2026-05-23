import re
from functools import lru_cache

from transformers import pipeline

from triage_agent.schemas import UrgencyResult

ZERO_SHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

URGENCY_KEYWORDS = [
    # English
    r"\burgent\b",
    r"\basap\b",
    r"\bemergency\b",
    r"\bimmediately\b",
    r"\bcritical\b",
    r"\bdeadline\b",
    r"\bstolen\b",
    r"\baccident\b",
    r"\bhospital\b",
    r"\blocked out\b",
    # German
    r"\bdringend\b",
    r"\bsofort\b",
    r"\beilig\b",
    r"\bnotfall\b",
    r"\bfrist\b",
    r"\bgestohlen\b",
    r"\bunfall\b",
    r"\bkrankenhaus\b",
    r"\bgesperrt\b",
    r"\bschaden\b",
]

CANDIDATE_LABELS = ["urgent and critical", "general inquiry"]


@lru_cache(maxsize=1)
def _get_zero_shot():
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


def _signal_score(text: str) -> tuple[float, list[str]]:
    """Regex-based score from urgency keywords. Returns (score, matched_terms)."""
    text_lower = text.lower()
    matched = []
    for pattern in URGENCY_KEYWORDS:
        m = re.search(pattern, text_lower)
        if m:
            matched.append(m.group(0))
    # Saturating score: 1 match = 0.5, 2+ matches = 1.0
    score = min(len(matched) * 0.5, 1.0)
    return score, matched


def _zero_shot_score(text: str) -> float:
    classifier = _get_zero_shot()
    result = classifier(text, CANDIDATE_LABELS)
    # `result["labels"]` is sorted by score; find the "urgent..." label's score
    label_to_score = dict(zip(result["labels"], result["scores"]))
    return float(label_to_score[CANDIDATE_LABELS[0]])


def score_urgency(text: str) -> UrgencyResult:
    """Hybrid urgency scoring: keyword signals + zero-shot classification."""
    signal_score, matched = _signal_score(text)
    zs_score = _zero_shot_score(text)
    final = 0.5 * signal_score + 0.5 * zs_score

    if final >= 0.66:
        level = "high"
    elif final >= 0.33:
        level = "medium"
    else:
        level = "low"

    return UrgencyResult(level=level, score=final, signals_found=matched)
