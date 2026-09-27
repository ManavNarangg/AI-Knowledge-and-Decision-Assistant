"""System 1 decision models - Boolean, Choice and Score evaluators.

Each evaluator receives one typed DecisionSpec plus the visitor's facts and returns a structured
result. Evaluation is deterministic, uses three-valued logic (TRUE / FALSE / UNKNOWN when a
required fact was not provided) and never calls an LLM. Decisions are independent, so they are
executed in parallel and a failure in one is isolated from the others.
"""
import asyncio
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

from ..schemas import Condition, DecisionSpec, Fact


class Tri(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class DecisionModelError(Exception):
    """Raised when a decision spec is malformed and cannot be evaluated."""


TRUE_WORDS = {"true", "yes", "y", "1", "has", "have"}
FALSE_WORDS = {"false", "no", "n", "0", "none", "not"}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def to_number(value: str) -> float | None:
    cleaned = re.sub(r"[,$€£%\s]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:st|nd|rd|th)?", cleaned.lower())
        return float(match.group(1)) if match else None


def to_bool(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v in TRUE_WORDS:
        return True
    if v in FALSE_WORDS:
        return False
    return None


LIST_SPLIT = re.compile(r"\s*(?:,|;|\bor\b|\band\b)\s*", re.IGNORECASE)


def split_list(value: str) -> list[str]:
    """'warehouse operations, fleet maintenance and dispatch' -> ['warehouse operations', ...]."""
    return [v.strip().lower() for v in LIST_SPLIT.split(value) if v and v.strip()]


def _looks_like_list(value: str) -> bool:
    return len(split_list(value)) > 1


def _matches(fact_value: str, item: str) -> bool:
    """Exact match, or one names the other as a whole word ('dispatch' vs 'dispatch team')."""
    f = fact_value.strip().lower()
    if not f or not item:
        return False
    return (f == item or re.search(rf"\b{re.escape(f)}\b", item) is not None
            or re.search(rf"\b{re.escape(item)}\b", f) is not None)


@dataclass
class ConditionResult:
    description: str
    fact: str
    operator: str
    expected: str
    actual: str | None
    result: Tri
    source: str


def evaluate_condition(cond: Condition, facts: dict[str, Fact]) -> ConditionResult:
    name = normalize_name(cond.fact)
    fact = facts.get(name)
    out = ConditionResult(cond.description, name, cond.operator, cond.value,
                          fact.value if fact else None, Tri.UNKNOWN, cond.source)
    if fact is None:
        return out

    op = cond.operator
    if op in ("is_true", "is_false"):
        b = to_bool(fact.value)
        if b is None:
            raise DecisionModelError(f"Fact '{name}'='{fact.value}' is not a yes/no value")
        out.result = Tri.TRUE if b == (op == "is_true") else Tri.FALSE
        return out

    left, right = to_number(fact.value), to_number(cond.value)

    # A text equality against a list ("warehouse operations, fleet maintenance and dispatch") is a
    # membership test in disguise; LLMs sometimes emit it that way, so normalize it here.
    if op in ("==", "!=") and (left is None or right is None) and _looks_like_list(cond.value):
        op = "in" if op == "==" else "not_in"

    if op in ("in", "not_in"):
        allowed = split_list(cond.value)
        if not allowed:
            raise DecisionModelError(f"Condition '{cond.description}' has an empty value list")
        hit = any(_matches(fact.value, item) for item in allowed)
        out.result = Tri.TRUE if hit == (op == "in") else Tri.FALSE
        return out

    if left is not None and right is not None:
        ok = {">=": left >= right, ">": left > right, "<=": left <= right,
              "<": left < right, "==": left == right, "!=": left != right}[op]
    elif op in ("==", "!="):
        lb, rb = to_bool(fact.value), to_bool(cond.value)
        same = (lb == rb) if lb is not None and rb is not None \
            else fact.value.strip().lower() == cond.value.strip().lower()
        ok = same if op == "==" else not same
    else:
        raise DecisionModelError(
            f"Cannot compare '{fact.value}' {op} '{cond.value}' numerically for '{cond.description}'")
    out.result = Tri.TRUE if ok else Tri.FALSE
    return out


def tri_all(values: list[Tri]) -> Tri:
    """Kleene AND: any FALSE -> FALSE; else any UNKNOWN -> UNKNOWN; else TRUE."""
    if Tri.FALSE in values:
        return Tri.FALSE
    return Tri.UNKNOWN if Tri.UNKNOWN in values else Tri.TRUE


def tri_any(values: list[Tri]) -> Tri:
    """Kleene OR: any TRUE -> TRUE; else any UNKNOWN -> UNKNOWN; else FALSE."""
    if Tri.TRUE in values:
        return Tri.TRUE
    return Tri.UNKNOWN if Tri.UNKNOWN in values else Tri.FALSE


@dataclass
class DecisionResult:
    id: str
    type: str
    question: str
    outcome: str | float | None = None          # "true"/"false"/"unknown", option label, or score
    band: str | None = None                     # score only: strong / moderate / weak
    detail: str = ""
    conditions: list[dict] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _pack(results: list[ConditionResult]) -> tuple[list[dict], list[str], list[str]]:
    conds = [{**asdict(r), "result": r.result.value} for r in results]
    missing = sorted({r.fact for r in results if r.result == Tri.UNKNOWN})
    sources = sorted({r.source for r in results if r.source})
    return conds, missing, sources


def evaluate_boolean(spec: DecisionSpec, facts: dict[str, Fact]) -> DecisionResult:
    if not spec.conditions:
        raise DecisionModelError(f"Boolean decision {spec.id} has no conditions")
    results = [evaluate_condition(c, facts) for c in spec.conditions]
    outcome = (tri_all if spec.match == "all" else tri_any)([r.result for r in results])
    conds, missing, sources = _pack(results)
    met = sum(r.result == Tri.TRUE for r in results)
    return DecisionResult(spec.id, "boolean", spec.question, outcome.value,
                          detail=f"{met}/{len(results)} conditions met (match={spec.match})",
                          conditions=conds, missing_facts=missing if outcome == Tri.UNKNOWN else [],
                          sources=sources)


def evaluate_choice(spec: DecisionSpec, facts: dict[str, Fact]) -> DecisionResult:
    """Selects the first option (priority order) whose conditions all hold.

    If an earlier option is UNKNOWN the choice cannot be settled yet, so the result is
    'unknown' and the facts needed to settle it are reported.
    """
    if not spec.options:
        raise DecisionModelError(f"Choice decision {spec.id} has no options")
    all_results, missing, blocked_by = [], set(), None
    selected = None
    for option in spec.options:
        results = [evaluate_condition(c, facts) for c in option.conditions]
        all_results.extend(results)
        verdict = tri_all([r.result for r in results]) if results else Tri.TRUE
        if verdict == Tri.TRUE and blocked_by is None:
            selected = option.label
            break
        if verdict == Tri.UNKNOWN and blocked_by is None:
            blocked_by = option.label
            missing.update(r.fact for r in results if r.result == Tri.UNKNOWN)
    conds, _, sources = _pack(all_results)
    if selected:
        return DecisionResult(spec.id, "choice", spec.question, selected,
                              detail=f"Selected '{selected}'", conditions=conds, sources=sources)
    if blocked_by:
        return DecisionResult(spec.id, "choice", spec.question, "unknown",
                              detail=f"Cannot decide whether '{blocked_by}' applies",
                              conditions=conds, missing_facts=sorted(missing), sources=sources)
    return DecisionResult(spec.id, "choice", spec.question, "none",
                          detail="No option's conditions are satisfied", conditions=conds, sources=sources)


def evaluate_score(spec: DecisionSpec, facts: dict[str, Fact]) -> DecisionResult:
    """Weighted support score in [0, 1]; UNKNOWN criteria are reported as a range."""
    if not spec.criteria:
        raise DecisionModelError(f"Score decision {spec.id} has no criteria")
    weights = [max(0.0, float(c.weight)) for c in spec.criteria]
    total = sum(weights)
    if total <= 0:
        raise DecisionModelError(f"Score decision {spec.id} has zero total weight")
    results = [evaluate_condition(c.condition, facts) for c in spec.criteria]
    got = sum(w for w, r in zip(weights, results) if r.result == Tri.TRUE) / total
    unknown = sum(w for w, r in zip(weights, results) if r.result == Tri.UNKNOWN) / total
    band = "strong" if got >= 0.75 else "moderate" if got >= 0.4 else "weak"
    conds, missing, sources = _pack(results)
    detail = f"score {got:.2f}" + (f" (could reach {got + unknown:.2f} with missing facts)" if unknown else "")
    return DecisionResult(spec.id, "score", spec.question, round(got, 3), band=band, detail=detail,
                          conditions=conds, missing_facts=missing, sources=sources)


EVALUATORS = {"boolean": evaluate_boolean, "choice": evaluate_choice, "score": evaluate_score}


def build_fact_table(facts: list[Fact]) -> dict[str, Fact]:
    return {normalize_name(f.name): f for f in facts}


def _run_one(spec: DecisionSpec, facts: dict[str, Fact]) -> DecisionResult:
    start = time.perf_counter()
    try:
        result = EVALUATORS[spec.type](spec, facts)
    except Exception as exc:  # isolate: one bad decision must not sink the others
        result = DecisionResult(spec.id, spec.type, spec.question, None,
                                error=f"{type(exc).__name__}: {exc}")
    result.elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    return result


async def run_decisions(specs: list[DecisionSpec], facts: list[Fact]) -> list[DecisionResult]:
    """Evaluate independent decisions concurrently."""
    table = build_fact_table(facts)
    return list(await asyncio.gather(*(asyncio.to_thread(_run_one, s, table) for s in specs)))
