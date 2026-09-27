"""Question-processing pipeline (application logic / orchestration).

 Visitor question
   -> [tenant]      assistant token -> organization (done by the API layer)
   -> [System 1]    retrieval (BM25, org-scoped)  ||  question-type classifier     (parallel)
   -> [System 1]    evidence-sufficiency score  -- "none" -> answer "not in KB", no LLM call
   -> [App logic]   route: does this question need structured decisions?
   -> [LLM]         decompose question + policy text into typed facts & decision specs
   -> [System 1]    Boolean / Choice / Score evaluators                            (parallel)
   -> [App logic]   combine results (three-valued logic), detect missing facts / failures
   -> [LLM]         write grounded answer from passages + authoritative decision results
   -> answer + page-level references + trace
"""
import asyncio
import re
import time
import uuid

from .combiner import combine, split_compound_decisions
from .db import get_conn, utcnow
from .llm import LLMError, get_llm
from .retrieval import retrieve
from .schemas import Decomposition
from .system1.classifier import classifier
from .system1.decisions import run_decisions
from .system1.evidence import assess

DECISION_ROUTES = {"decision", "comparison"}
ROUTE_CONFIDENCE = 0.55
TOP_K = 6
TOP_K_COMPARISON = 8

NOT_FOUND_MESSAGE = ("I couldn't find information about this in {org}'s knowledge base, so I can't "
                     "answer it reliably. Please contact {org} directly for help with this question.")
FACTS_NEEDED_NOTE = ("One or more required checks could not be evaluated. Do not give a yes/no verdict; "
                     "explain what the policy requires and recommend contacting the organization.")


SMALL_TALK = re.compile(
    r"^\s*(?:(?P<greet>hi+|hello|hey|hiya|yo|good\s+(?:morning|afternoon|evening))|"
    r"(?P<thanks>thanks|thank\s+you|thx|ty|cheers)|(?P<bye>bye|goodbye|see\s+you))"
    r"(?:\s+(?:there|so\s+much|a\s+lot))?[\s!.?,:)]*$", re.IGNORECASE)


def small_talk_reply(question: str, org_name: str) -> str | None:
    """Greetings/thanks aren't knowledge questions: answer them politely without retrieval or an LLM."""
    m = SMALL_TALK.match(question)
    if not m:
        return None
    if m.group("thanks"):
        return "You're welcome! Let me know if you have any other questions."
    if m.group("bye"):
        return "Goodbye! Come back any time you have a question."
    return (f"Hi! I'm {org_name}'s assistant. Ask me anything about its documents and policies - "
            "you can include your own details (like year, GPA or tenure) and I'll check them against the rules.")


class Stopwatch:
    def __init__(self):
        self.stages: list[dict] = []

    def add(self, name: str, component: str, start: float, **info) -> None:
        self.stages.append({"stage": name, "component": component,
                            "ms": round((time.perf_counter() - start) * 1000, 1), **info})


def _decision_context(decomposition: Decomposition, results, combined) -> dict:
    ctx = {
        "overall_outcome": combined.overall,
        "combination_rule": decomposition.combine,
        "visitor_facts": {f.name: f.value for f in decomposition.facts},
        "missing_facts": combined.missing_facts,
        "decisions": [
            {"question": r.question, "type": r.type, "outcome": r.outcome, "band": r.band,
             "detail": r.detail, "error": r.error,
             "checks": [{"rule": c["description"], "visitor_value": c["actual"], "required": f'{c["operator"]} {c["expected"]}'.strip(),
                         "result": c["result"], "source": c["source"]} for c in r.conditions]}
            for r in results
        ],
    }
    if combined.overall == "error":
        ctx["note"] = FACTS_NEEDED_NOTE
    return ctx


def _references(passages, cited_ids: set[str]) -> list[dict]:
    """One reference per (document, page); keeps every source id that points at it."""
    by_page: dict[tuple, dict] = {}
    for p in passages:
        if p.sid not in cited_ids:
            continue
        key = (p.filename, p.page)
        if key in by_page:
            by_page[key]["source_ids"].append(p.sid)
        else:
            by_page[key] = {**p.reference(), "source_ids": [p.sid]}
    return list(by_page.values())


def _log(org_id: str, question: str, status: str, qtype: str | None, latency_ms: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO questions (id, org_id, question, status, question_type, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), org_id, question[:1000], status, qtype, latency_ms, utcnow()),
        )


async def answer_question(org_id: str, org_name: str, question: str) -> dict:
    started = time.perf_counter()
    sw = Stopwatch()
    trace: dict = {"organization": org_name, "llm_calls": 0}

    def finish(status: str, answer: str, references: list[dict], error: str | None = None) -> dict:
        latency = int((time.perf_counter() - started) * 1000)
        trace["stages"] = sw.stages
        _log(org_id, question, status, trace.get("classification", {}).get("label"), latency)
        return {"status": status, "answer": answer, "references": references,
                "error": error, "latency_ms": latency, "trace": trace}

    t = time.perf_counter()
    reply = small_talk_reply(question, org_name)
    if reply:
        sw.add("small-talk check", "App logic", t, matched=True)
        trace["classification"] = {"label": "small_talk", "confidence": 1.0, "probabilities": {}}
        return finish("answered", reply, [])

    # 1. Retrieval and classification are independent System 1 steps: run them in parallel.
    t = time.perf_counter()
    retrieval, classification = await asyncio.gather(
        asyncio.to_thread(retrieve, org_id, question, TOP_K_COMPARISON),
        asyncio.to_thread(classifier.predict, question),
    )
    if classification.label != "comparison":
        retrieval.passages = retrieval.passages[:TOP_K]
        retrieval.passage_terms = retrieval.passage_terms[:TOP_K]
    passages = retrieval.passages
    sw.add("retrieval + classification (parallel)", "System 1", t)
    trace["classification"] = {"label": classification.label, "confidence": classification.confidence,
                               "probabilities": classification.probabilities}
    trace["retrieval"] = [{"source_id": p.sid, "document": p.filename, "page": p.page, "bm25": p.score}
                          for p in passages]

    # 2. Evidence gate: refuse early rather than let an LLM improvise.
    t = time.perf_counter()
    evidence = assess(retrieval)
    sw.add("evidence sufficiency score", "System 1", t, score=evidence.score, verdict=evidence.verdict)
    trace["evidence"] = evidence.__dict__
    if evidence.verdict == "none":
        return finish("unable_to_answer", NOT_FOUND_MESSAGE.format(org=org_name), [])

    llm = get_llm()
    degraded_reasons: list[str] = []
    decision_context = None
    decision_sources: set[str] = set()

    # 3. Routing: only decision-shaped (or uncertain) questions pay for decomposition.
    needs_decisions = classification.label in DECISION_ROUTES or classification.confidence < ROUTE_CONFIDENCE
    trace["route"] = {"decomposition": needs_decisions,
                      "reason": (f"classified as '{classification.label}'" if classification.label in DECISION_ROUTES
                                 else f"low classifier confidence ({classification.confidence:.2f})" if needs_decisions
                                 else f"'{classification.label}' question answered directly from passages")}

    if needs_decisions:
        t = time.perf_counter()
        try:
            trace["llm_calls"] += 1
            decomposition = await asyncio.to_thread(llm.decompose, question, passages, classification.label)
            sw.add("question decomposition", "LLM", t, decisions=len(decomposition.decisions))
            trace["decomposition"] = decomposition.model_dump()
        except LLMError as exc:
            sw.add("question decomposition", "LLM", t, error=str(exc))
            degraded_reasons.append("decomposition failed")
            decomposition = None

        if decomposition and decomposition.decisions:
            t = time.perf_counter()
            decomposition = split_compound_decisions(decomposition)      # one decision per requirement
            results = await run_decisions(decomposition.decisions, decomposition.facts)
            sw.add(f"{len(results)} decision model(s) (parallel)", "System 1", t)
            trace["decisions"] = [r.to_dict() for r in results]

            t = time.perf_counter()
            combined = combine(decomposition, results)
            sw.add("combine results", "App logic", t, overall=combined.overall)
            trace["combined"] = combined.to_dict()
            if combined.needs_review:
                degraded_reasons.append("decision model failure")
            decision_context = _decision_context(decomposition, results, combined)
            decision_sources = {s for r in results for s in r.sources}

    # 4. Grounded answer generation.
    t = time.perf_counter()
    try:
        trace["llm_calls"] += 1
        generated = await asyncio.to_thread(llm.generate, question, passages, decision_context)
        sw.add("answer generation", "LLM", t, grounded=generated.grounded)
    except LLMError as exc:
        sw.add("answer generation", "LLM", t, error=str(exc))
        excerpts = "\n\n".join(
            f"> {' '.join(p.text.split())[:400]}{'…' if len(p.text) > 400 else ''} [{p.sid}]" for p in passages[:3])
        answer = ("The AI answer service is temporarily unavailable, so I can't compose a full answer right now. "
                  f"These are the most relevant passages from the knowledge base:\n\n{excerpts}")
        return finish("error", answer, [p.reference() for p in passages[:3]], error=str(exc))

    if not generated.grounded:
        return finish("unable_to_answer", NOT_FOUND_MESSAGE.format(org=org_name), [])

    valid_ids = {p.sid for p in passages}
    cited = (set(generated.sources_used) | set(re.findall(r"\[(S\d+)\]", generated.answer))) & valid_ids
    if decision_context:
        cited |= decision_sources & valid_ids
    status = "needs_review" if degraded_reasons else "answered"
    if degraded_reasons:
        trace["review_reasons"] = degraded_reasons
    return finish(status, generated.answer, _references(passages, cited))
