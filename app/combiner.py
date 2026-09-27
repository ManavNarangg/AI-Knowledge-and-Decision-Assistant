"""Application logic: combine independent decision results into one overall result.

This is ordinary, auditable code - neither the LLM nor a model decides how results combine.
"""
from dataclasses import asdict, dataclass, field

from .schemas import Decomposition
from .system1.decisions import DecisionResult, Tri, tri_all, tri_any


@dataclass
class CombinedResult:
    overall: str                     # yes | no | insufficient_information | informational | error
    summary: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    failed_decisions: list[str] = field(default_factory=list)
    needs_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def split_compound_decisions(decomposition: Decomposition) -> Decomposition:
    """Make every requirement its own independent decision.

    LLMs often bundle several requirements into one boolean decision ("all of: GPA, year, ...").
    When the overall rule is "all" that is logically identical to one decision per condition,
    so we split them: each requirement is then evaluated independently (in parallel), reported
    separately, and a failure in one is isolated from the rest.
    """
    if decomposition.combine != "all":
        return decomposition
    atomic = []
    for spec in decomposition.decisions:
        if spec.type == "boolean" and spec.match == "all" and len(spec.conditions) > 1:
            atomic.extend(
                spec.model_copy(update={"id": f"{spec.id}.{i}", "conditions": [cond],
                                        "question": cond.description or f"{cond.fact} {cond.operator} {cond.value}"})
                for i, cond in enumerate(spec.conditions, start=1))
        else:
            atomic.append(spec)
    return decomposition.model_copy(update={"decisions": atomic})


def combine(decomposition: Decomposition, results: list[DecisionResult]) -> CombinedResult:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    summary = []
    for r in results:
        if r.error:
            summary.append(f"{r.id} ({r.type}) could not be evaluated: {r.error}")
        elif r.type == "score":
            summary.append(f"{r.id} score: {r.question} -> {r.outcome} ({r.band}); {r.detail}")
        else:
            summary.append(f"{r.id} {r.type}: {r.question} -> {r.outcome}; {r.detail}")

    booleans = [Tri(r.outcome) for r in ok if r.type == "boolean"]
    if failed and decomposition.combine != "none" and any(r.type == "boolean" for r in failed):
        overall = "error"            # a required yes/no check failed: we must not guess
    elif decomposition.combine == "none" or not booleans:
        choices = [r for r in ok if r.type == "choice"]
        overall = "insufficient_information" if any(c.outcome == "unknown" for c in choices) else "informational"
    else:
        verdict = (tri_all if decomposition.combine == "all" else tri_any)(booleans)
        overall = {Tri.TRUE: "yes", Tri.FALSE: "no", Tri.UNKNOWN: "insufficient_information"}[verdict]

    # Missing facts matter only when they could still change the overall outcome.
    missing = sorted({m for r in ok for m in r.missing_facts}) if overall != "no" else []
    return CombinedResult(
        overall=overall,
        summary=summary,
        missing_facts=missing,
        failed_decisions=[r.id for r in failed],
        needs_review=bool(failed),
    )
