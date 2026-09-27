"""Unit tests for the System 1 models and the combiner."""
import asyncio
import json

from app.combiner import combine
from app.schemas import Condition, Criterion, Decomposition, DecisionSpec, Fact, Option
from app.system1.classifier import classifier
from app.system1.decisions import Tri, build_fact_table, evaluate_condition, run_decisions

from .conftest import SAMPLE


def cond(fact, op, value="", source="S1"):
    return Condition(fact=fact, operator=op, value=value, description=f"{fact} {op} {value}", source=source)


def boolean(id_, *conditions, match="all"):
    return DecisionSpec(id=id_, type="boolean", question=id_, match=match,
                        conditions=list(conditions), options=[], criteria=[])


FACTS = [Fact(name="GPA", value="3.72", kind="number"),
         Fact(name="year_of_study", value="2", kind="number"),
         Fact(name="has_sports_scholarship", value="true", kind="boolean"),
         Fact(name="sports_scholarship_amount", value="$4,000", kind="number"),
         Fact(name="department", value="Dispatch", kind="text")]


def test_condition_operators_and_normalization():
    table = build_fact_table(FACTS)
    assert evaluate_condition(cond("gpa", ">=", "3.5"), table).result == Tri.TRUE
    assert evaluate_condition(cond("Year of Study", ">=", "2"), table).result == Tri.TRUE
    assert evaluate_condition(cond("sports_scholarship_amount", "<", "5000"), table).result == Tri.TRUE
    assert evaluate_condition(cond("has_sports_scholarship", "is_false"), table).result == Tri.FALSE
    assert evaluate_condition(cond("department", "not_in", "warehouse, fleet maintenance, dispatch"), table).result == Tri.FALSE
    assert evaluate_condition(cond("household_income", "<", "60000"), table).result == Tri.UNKNOWN


def test_three_valued_boolean_and_parallel_execution():
    specs = [boolean("gpa", cond("gpa", ">=", "3.5")),
             boolean("income", cond("household_income", "<", "60000")),
             boolean("sports", cond("sports_scholarship_amount", ">=", "5000"))]
    results = asyncio.run(run_decisions(specs, FACTS))
    assert [r.outcome for r in results] == ["true", "unknown", "false"]
    assert results[1].missing_facts == ["household_income"]


def test_choice_selects_tier_in_priority_order():
    tiers = DecisionSpec(id="tier", type="choice", question="Which tier?", match="all", conditions=[], criteria=[], options=[
        Option(label="Gold", conditions=[cond("gpa", ">=", "3.85")]),
        Option(label="Silver", conditions=[cond("gpa", ">=", "3.70")]),
        Option(label="Bronze", conditions=[cond("gpa", ">=", "3.50")]),
    ])
    [r] = asyncio.run(run_decisions([tiers], FACTS))
    assert r.outcome == "Silver"
    [r] = asyncio.run(run_decisions([tiers], []))
    assert r.outcome == "unknown" and r.missing_facts == ["gpa"]


def test_score_reports_band_and_range():
    spec = DecisionSpec(id="s", type="score", question="support?", match="all", conditions=[], options=[], criteria=[
        Criterion(condition=cond("gpa", ">=", "3.5"), weight=0.5),
        Criterion(condition=cond("year_of_study", ">=", "2"), weight=0.3),
        Criterion(condition=cond("household_income", "<", "60000"), weight=0.2),
    ])
    [r] = asyncio.run(run_decisions([spec], FACTS))
    assert r.outcome == 0.8 and r.band == "strong" and "could reach 1.00" in r.detail


def test_malformed_decision_is_isolated():
    bad = boolean("bad", cond("department", ">=", "5"))          # text vs number -> model error
    good = boolean("good", cond("gpa", ">=", "3.5"))
    empty = DecisionSpec(id="empty", type="choice", question="?", match="all", conditions=[], options=[], criteria=[])
    results = asyncio.run(run_decisions([bad, good, empty], FACTS))
    assert results[0].error and results[2].error and results[1].outcome == "true"
    combined = combine(Decomposition(restated_question="", facts=FACTS, decisions=[bad, good, empty], combine="all"), results)
    assert combined.overall == "error" and combined.needs_review and combined.failed_decisions == ["bad", "empty"]


def test_combiner_logic():
    d = lambda combine_rule: Decomposition(restated_question="", facts=[], decisions=[], combine=combine_rule)  # noqa: E731
    t, f, u = (boolean(x, cond("gpa", ">=", v)) for x, v in [("t", "3"), ("f", "4"), ("u", "0")])
    facts = [Fact(name="gpa", value="3.6", kind="number")]
    rt, rf = asyncio.run(run_decisions([t, f], facts))
    [ru] = asyncio.run(run_decisions([boolean("u", cond("income", "<", "1"))], facts))
    assert combine(d("all"), [rt, rf]).overall == "no"
    assert combine(d("any"), [rt, rf]).overall == "yes"
    assert combine(d("all"), [rt, ru]).overall == "insufficient_information"
    assert combine(d("all"), [rt, ru]).missing_facts == ["income"]
    assert combine(d("all"), [rf, ru]).overall == "no"          # FALSE dominates UNKNOWN
    assert combine(d("none"), [rt]).overall == "informational"


def test_classifier_on_sample_questions():
    expected = {"NB-1": "factual", "NB-3": "procedural", "NB-4": "decision", "NB-5": "decision",
                "NB-7": "decision", "NB-8": "comparison", "AC-2": "decision", "AC-4": "procedural", "AC-5": "comparison"}
    cases = {c["id"]: c["question"] for c in json.loads((SAMPLE / "test_questions.json").read_text())}
    wrong = {k: classifier.predict(cases[k]).label for k in expected if classifier.predict(cases[k]).label != expected[k]}
    assert not wrong, wrong


def test_retrieval_matches_abbreviation_and_typo_variants():
    from app.retrieval import BM25Index, _one_edit
    rows = [{"id": 1, "filename": "resume.pdf", "text": "Education: B.Tech, CGPA: 8.89/10"},
            {"id": 2, "filename": "resume.pdf", "text": "Projects: built a scholarship portal"}]
    index = BM25Index(rows)
    assert index.expand("gpa") == {"cgpa": 0.8}
    assert index.expand("scholarshp") == {"scholarship": 0.7}
    assert index.expand("menu") == {}
    assert index.search("what is the gpa", 1)[0][0]["id"] == 1
    assert _one_edit("color", "colour") and not _one_edit("policy", "police x")


def test_list_rules_written_as_equality_are_treated_as_membership():
    # Real Gemini output for AC-2: "role != <whole list>" must NOT pass for a dispatcher.
    table = build_fact_table([Fact(name="role", value="dispatch", kind="text")])
    rule = "warehouse operations, fleet maintenance and dispatch"
    assert evaluate_condition(cond("role", "!=", rule), table).result == Tri.FALSE
    assert evaluate_condition(cond("role", "not_in", rule), table).result == Tri.FALSE
    assert evaluate_condition(cond("role", "==", rule), table).result == Tri.TRUE
    other = build_fact_table([Fact(name="role", value="Finance", kind="text")])
    assert evaluate_condition(cond("role", "not_in", rule), other).result == Tri.TRUE
    assert evaluate_condition(cond("role", "in", "Technology, Finance or Human Resources"), other).result == Tri.TRUE


def test_bundled_decision_is_split_into_independent_requirements():
    from app.combiner import split_compound_decisions
    bundled = Decomposition(restated_question="", facts=[], combine="all", decisions=[
        boolean("D1", cond("gpa", ">=", "3.5"), cond("year_of_study", "in", "2,3,4"), cond("role", "!=", "x")),
        DecisionSpec(id="D2", type="choice", question="tier", options=[Option(label="A", conditions=[])])])
    split = split_compound_decisions(bundled)
    assert [d.id for d in split.decisions] == ["D1.1", "D1.2", "D1.3", "D2"]
    assert all(len(d.conditions) == 1 for d in split.decisions[:3])
    # "any" semantics would change if split, so those are left alone.
    anyd = bundled.model_copy(update={"combine": "any"})
    assert split_compound_decisions(anyd).decisions == anyd.decisions
