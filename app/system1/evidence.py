"""System 1 model #2 - Evidence Sufficiency Score (answerability gate).

Scores 0-1 how well the retrieved passages cover the informative (IDF-weighted) terms of the
question. Terms that never occur in the organization's knowledge base get maximal IDF, so a
question about something the documents never mention scores near zero and is answered with
"not in the knowledge base" WITHOUT any LLM call - the system cannot hallucinate what it never
asks an LLM to write.

This is the first of two grounding layers; the answer-generation LLM independently reports
`grounded=false` when passages exist but don't actually answer the question.
"""
from dataclasses import dataclass

from ..retrieval import RetrievalResult, stem

# Words that describe the *shape* of a question or the visitor's situation rather than its topic.
# They carry no evidence either way, so they are excluded from coverage.
QUESTION_PHRASING = {stem(w) for w in """
difference differences differ different compare comparison versus vs between better worse which
many much already currently next last previous rule rules entitled eligible eligibility qualify
first second third fourth fifth year years month months week weeks day days
""".split()}

NO_EVIDENCE_THRESHOLD = 0.30
WEAK_EVIDENCE_THRESHOLD = 0.5


@dataclass
class EvidenceAssessment:
    score: float
    verdict: str                 # "none" | "weak" | "sufficient"
    best_passage_coverage: float
    combined_coverage: float
    uncovered_terms: list[str]


def assess(result: RetrievalResult, top_n: int = 3) -> EvidenceAssessment:
    terms = sorted({t for t in result.query_terms if not t[0].isdigit() and t not in QUESTION_PHRASING})
    if not terms or not result.passages:
        return EvidenceAssessment(0.0, "none", 0.0, 0.0, terms)

    weights = {t: result.index.idf(t) for t in terms}
    total = sum(weights.values()) or 1.0

    variants = {t: result.index.expand(t) or {t: 1.0} for t in terms}

    def covered(t, passage_terms) -> float:
        """Match strength of term t in a passage (1.0 exact, <1 for an abbreviation/typo variant)."""
        return max((w for v, w in variants[t].items() if v in passage_terms), default=0.0)

    def coverage(term_sets) -> float:
        return sum(w * max(covered(t, s) for s in term_sets) for t, w in weights.items()) / total

    best = max(coverage([s]) for s in result.passage_terms[:top_n])
    combined = coverage(result.passage_terms[:top_n])
    score = round(0.5 * best + 0.5 * combined, 3)
    uncovered = [t for t in terms if not any(covered(t, s) for s in result.passage_terms[:top_n])]

    if score < NO_EVIDENCE_THRESHOLD:
        verdict = "none"
    elif score < WEAK_EVIDENCE_THRESHOLD:
        verdict = "weak"
    else:
        verdict = "sufficient"
    return EvidenceAssessment(score, verdict, round(best, 3), round(combined, 3), uncovered)
