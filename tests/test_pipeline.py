"""End-to-end pipeline tests through POST /api/ask with a scripted LLM."""
from app.schemas import Condition, Decomposition, DecisionSpec, Fact, Option

from .conftest import new_client

MULTI = ("I'm a second-year full-time student with a 3.72 GPA and no misconduct record. I already have a "
         "sports scholarship worth $4,000 a year. Am I eligible for the Merit Scholarship, and which tier would I get?")


def c(fact, op, value, source):
    return Condition(fact=fact, operator=op, value=value, description=f"{fact} {op} {value}", source=source)


def b(id_, *conds):
    return DecisionSpec(id=id_, type="boolean", question=id_, match="all", conditions=list(conds), options=[], criteria=[])


def merit_decomposition(sports_amount: str) -> Decomposition:
    """What the LLM is expected to produce for the merit question: facts + independent typed decisions."""
    return Decomposition(
        restated_question="Merit eligibility and tier",
        facts=[Fact(name="enrollment_credits_full_time", value="true", kind="boolean"),
               Fact(name="year_of_study", value="2", kind="number"),
               Fact(name="gpa", value="3.72", kind="number"),
               Fact(name="has_misconduct_record", value="false", kind="boolean"),
               Fact(name="sports_scholarship_amount", value=sports_amount, kind="number")],
        decisions=[
            b("D1", c("enrollment_credits_full_time", "is_true", "", "S1")),
            b("D2", c("year_of_study", "in", "2,3,4", "S1")),
            b("D3", c("gpa", ">=", "3.5", "S1")),
            b("D4", c("has_misconduct_record", "is_false", "", "S1")),
            b("D5", c("sports_scholarship_amount", "<", "5000", "S2")),
            DecisionSpec(id="D6", type="choice", question="Which tier?", match="all", conditions=[], criteria=[], options=[
                Option(label="Gold ($7,500)", conditions=[c("gpa", ">=", "3.85", "S3")]),
                Option(label="Silver ($6,000)", conditions=[c("gpa", ">=", "3.70", "S3")]),
                Option(label="Bronze ($5,000)", conditions=[c("gpa", ">=", "3.50", "S3")])]),
        ],
        combine="all",
    )


def ask(two_orgs, question, org="a"):
    r = new_client().post("/api/ask", json={"token": two_orgs[f"token_{org}"], "question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_multi_decision_question_eligible(two_orgs, fake_llm):
    fake_llm.decompositions[MULTI] = merit_decomposition("4000")
    out = ask(two_orgs, MULTI)
    t = out["trace"]
    assert t["classification"]["label"] == "decision" and t["route"]["decomposition"]
    assert len(t["decisions"]) == 6 and all(d["error"] is None for d in t["decisions"])
    assert t["combined"]["overall"] == "yes"
    assert next(d for d in t["decisions"] if d["id"] == "D6")["outcome"] == "Silver ($6,000)"
    assert t["llm_calls"] == 2 and out["status"] == "answered"
    assert fake_llm.calls[-1]["context"]["overall_outcome"] == "yes"     # LLM receives, not computes, the verdict
    assert out["references"] and all(r["page"] >= 1 for r in out["references"])
    components = [s["component"] for s in t["stages"]]
    assert components == ["System 1", "System 1", "LLM", "System 1", "App logic", "LLM"]


def test_multi_decision_question_not_eligible(two_orgs, fake_llm):
    fake_llm.decompositions[MULTI] = merit_decomposition("6000")
    out = ask(two_orgs, MULTI)
    assert out["trace"]["combined"]["overall"] == "no"
    assert next(d for d in out["trace"]["decisions"] if d["id"] == "D5")["outcome"] == "false"


def test_simple_question_skips_decomposition(two_orgs, fake_llm):
    out = ask(two_orgs, "How do I apply for the Merit Scholarship and what documents do I need?")
    assert out["trace"]["route"]["decomposition"] is False
    assert out["trace"]["llm_calls"] == 1 and [c["stage"] for c in fake_llm.calls] == ["generate"]
    assert out["status"] == "answered"


def test_out_of_scope_question_never_reaches_llm(two_orgs, fake_llm):
    out = ask(two_orgs, "What is on the cafeteria menu this Friday?")
    assert out["status"] == "unable_to_answer" and out["references"] == []
    assert "knowledge base" in out["answer"] and fake_llm.calls == []


def test_llm_reports_ungrounded(two_orgs, fake_llm):
    fake_llm.grounded = False
    out = ask(two_orgs, "What is the policy on late submissions for group projects abroad?")
    assert out["status"] == "unable_to_answer" and out["references"] == []


def test_llm_failure_returns_extractive_fallback(two_orgs, fake_llm):
    fake_llm.fail_generate = True
    out = ask(two_orgs, "What is the application deadline for the Merit Scholarship?")
    assert out["status"] == "error" and "temporarily unavailable" in out["answer"]
    assert out["references"] and out["error"]


def test_decomposition_failure_degrades_to_needs_review(two_orgs, fake_llm):
    fake_llm.fail_decompose = True
    out = ask(two_orgs, MULTI)
    assert out["status"] == "needs_review" and "decomposition failed" in out["trace"]["review_reasons"]


def test_decision_model_failure_flags_review(two_orgs, fake_llm):
    bad = merit_decomposition("4000")
    bad.decisions.append(b("D7", c("gpa", ">=", "not-a-number", "S1")))
    fake_llm.decompositions[MULTI] = bad
    out = ask(two_orgs, MULTI)
    assert out["status"] == "needs_review" and out["trace"]["combined"]["overall"] == "error"
    assert "note" in fake_llm.calls[-1]["context"]


def test_usage_counts(two_orgs, fake_llm):
    ask(two_orgs, "What is the daily meal limit for international travel?", org="b")
    usage = two_orgs["b"].get("/api/admin/usage").json()
    assert usage["total"] >= 1 and usage["recent"][0]["question"].startswith("What is the daily meal")


def test_small_talk_is_answered_without_retrieval_or_llm(two_orgs, fake_llm):
    out = ask(two_orgs, "Hi there!")
    assert out["status"] == "answered" and "Northbridge University" in out["answer"]
    assert fake_llm.calls == [] and out["trace"]["classification"]["label"] == "small_talk"
    assert ask(two_orgs, "Hi, what is the application deadline?")["trace"]["classification"]["label"] != "small_talk"
