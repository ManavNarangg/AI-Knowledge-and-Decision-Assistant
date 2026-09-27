"""Conversation memory: transcript, fact memory, follow-ups, and tenant isolation of conversations."""
from app.schemas import Condition, Decomposition, DecisionSpec, Fact

from .conftest import new_client

FIRST = "Am I eligible for the Merit Scholarship? I'm a third-year full-time student."


def c(fact, op, value, source="S1"):
    return Condition(fact=fact, operator=op, value=value, description=f"{fact} {op} {value}", source=source)


def merit_rules(*facts: Fact) -> Decomposition:
    return Decomposition(restated_question="Merit eligibility", facts=list(facts), combine="all", decisions=[
        DecisionSpec(id="D1", type="boolean", question="Merit requirements", match="all", conditions=[
            c("is_full_time", "is_true", ""), c("year_of_study", "in", "2,3,4"), c("gpa", ">=", "3.5")])])


def ask(token, question, conversation_id=None):
    body = {"token": token, "question": question}
    if conversation_id:
        body["conversation_id"] = conversation_id
    r = new_client().post("/api/ask", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_missing_fact_is_asked_for_then_completed_on_the_next_turn(two_orgs, fake_llm):
    token = two_orgs["token_a"]
    fake_llm.decompositions[FIRST] = merit_rules(Fact(name="is_full_time", value="true", kind="boolean"),
                                                 Fact(name="year_of_study", value="3", kind="number"))
    first = ask(token, FIRST)
    assert first["trace"]["combined"]["overall"] == "insufficient_information"
    assert first["trace"]["combined"]["missing_facts"] == ["gpa"]
    cid = first["conversation_id"]

    # Turn 2: the visitor only says "3.9". The LLM extracts just gpa; memory supplies the rest.
    fake_llm.decompositions["3.9"] = merit_rules(Fact(name="gpa", value="3.9", kind="number"))
    second = ask(token, "3.9", cid)
    t = second["trace"]
    assert second["conversation_id"] == cid and t["memory"]["resumed"]
    assert t["memory"]["awaiting_facts"]["missing_facts"] == ["gpa"]
    assert t["memory"]["search_used_conversation"] is True           # "3.9" alone matches nothing
    assert t["route"]["reason"] == "continuing a decision that was waiting for facts"
    assert t["combined"]["overall"] == "yes"                         # year + full-time came from memory
    decompose_call = next(x for x in fake_llm.calls if x["stage"] == "decompose" and x["question"] == "3.9")
    assert {f["name"] for f in decompose_call["known_facts"]} == {"is_full_time", "year_of_study"}
    assert "Visitor: " + FIRST in decompose_call["history"]


def test_unrelated_new_question_is_searched_on_its_own(two_orgs, fake_llm):
    token = two_orgs["token_a"]
    cid = ask(token, "What is the application deadline for the Merit Scholarship?")["conversation_id"]
    out = ask(token, "How do I appeal a decision?", cid)
    assert out["trace"]["memory"]["resumed"] and out["trace"]["memory"]["search_used_conversation"] is False
    assert all(r["document"] == "Student_Services_Guide.pdf" for r in out["references"])
    assert "application deadline" in fake_llm.calls[-1]["history"]     # still available to the LLM


def test_conversation_cannot_be_continued_by_another_organization(two_orgs, fake_llm):
    cid = ask(two_orgs["token_a"], "What is the application deadline for the Merit Scholarship?")["conversation_id"]
    out = ask(two_orgs["token_b"], "How do I submit an expense claim?", cid)
    assert out["conversation_id"] != cid                               # silently starts a fresh one
    assert out["trace"]["memory"]["resumed"] is False and out["trace"]["memory"]["history_messages"] == 0
    assert fake_llm.calls[-1]["history"] == ""


def test_transcript_endpoint_is_scoped_to_the_organization(two_orgs, fake_llm):
    cid = ask(two_orgs["token_a"], "Hi")["conversation_id"]
    ask(two_orgs["token_a"], "What is the application deadline for the Merit Scholarship?", cid)
    anon = new_client()
    r = anon.get(f"/api/assistant/{two_orgs['token_a']}/conversations/{cid}")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[3]["references"] and msgs[3]["status"] == "answered"
    assert anon.get(f"/api/assistant/{two_orgs['token_b']}/conversations/{cid}").status_code == 404
    assert anon.get(f"/api/assistant/{two_orgs['token_a']}/conversations/not-a-real-id").status_code == 404


def test_bogus_conversation_id_starts_a_new_conversation(two_orgs, fake_llm):
    out = ask(two_orgs["token_a"], "What is the application deadline?", "x" * 500)
    assert out["trace"]["memory"]["resumed"] is False and len(out["conversation_id"]) == 32
