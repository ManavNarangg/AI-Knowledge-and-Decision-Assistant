"""Multi-tenant isolation and access-control tests (assignment section 11)."""
from app.db import get_conn
from app.retrieval import retrieve

from .conftest import new_client, register

NB_FILES = {"Scholarship_Rules.pdf", "Academic_Policies.pdf", "Student_Services_Guide.pdf"}
ACME_FILES = {"Leave_Policy.pdf", "Remote_Work_Policy.pdf", "Expense_Reimbursement_Policy.pdf"}


def org_id_for(token: str) -> str:
    with get_conn() as conn:
        return conn.execute("SELECT id FROM organizations WHERE assistant_token = ?", (token,)).fetchone()[0]


def test_retrieval_only_returns_own_org_chunks(two_orgs):
    a, b = org_id_for(two_orgs["token_a"]), org_id_for(two_orgs["token_b"])
    for query in ["merit scholarship GPA requirement", "remote work performance rating", "leave days"]:
        assert {p.filename for p in retrieve(a, query).passages} <= NB_FILES
        assert {p.filename for p in retrieve(b, query).passages} <= ACME_FILES


def test_llm_only_sees_the_asking_orgs_documents(two_orgs, fake_llm):
    r = two_orgs["a"].post("/api/ask", json={"token": two_orgs["token_a"],
                                             "question": "What is the application deadline for the Merit Scholarship?"})
    assert r.status_code == 200
    assert fake_llm.calls and all(call["files"] <= NB_FILES for call in fake_llm.calls)
    assert {ref["document"] for ref in r.json()["references"]} <= NB_FILES


def test_other_orgs_assistant_cannot_answer_from_foreign_documents(two_orgs, fake_llm):
    # A Northbridge-only question asked to Acme's assistant: no Acme evidence -> refused, no LLM call.
    r = two_orgs["b"].post("/api/ask", json={"token": two_orgs["token_b"],
                                             "question": "What GPA do I need for the Merit Scholarship?"})
    body = r.json()
    assert body["status"] == "unable_to_answer"
    assert body["references"] == []
    assert fake_llm.calls == []


def test_assistant_token_cannot_access_admin_apis(two_orgs):
    token = two_orgs["token_a"]
    anon, forged = new_client(), new_client()
    forged.cookies.set("kda_session", token)       # assistant token presented as a session cookie
    for attempt in [
        anon.get("/api/admin/documents", headers={"Authorization": f"Bearer {token}"}),
        anon.get(f"/api/admin/documents?token={token}"),
        forged.get("/api/admin/documents"),
        anon.post("/api/admin/assistant/regenerate", headers={"X-Assistant-Token": token}),
        anon.get("/api/admin/usage", headers={"Authorization": f"Bearer {token}"}),
    ]:
        assert attempt.status_code == 401


def test_admin_apis_require_login():
    anon = new_client()
    assert anon.get("/api/admin/me").status_code == 401
    assert anon.post("/api/admin/documents", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}).status_code == 401


def test_org_cannot_touch_another_orgs_documents(two_orgs):
    a, b = two_orgs["a"], two_orgs["b"]
    a_doc = a.get("/api/admin/documents").json()["documents"][0]
    assert b.delete(f"/api/admin/documents/{a_doc['id']}").status_code == 404
    replace = b.post(f"/api/admin/documents/{a_doc['id']}/replace",
                     files={"file": ("x.pdf", b"%PDF-1.4 junk", "application/pdf")})
    assert replace.status_code in (404, 422)
    assert a_doc["id"] in {d["id"] for d in a.get("/api/admin/documents").json()["documents"]}
    assert all(d["filename"] in ACME_FILES for d in b.get("/api/admin/documents").json()["documents"])


def test_uploaded_pdfs_are_not_publicly_served(two_orgs):
    anon = new_client()
    a_org = org_id_for(two_orgs["token_a"])
    doc_id = two_orgs["a"].get("/api/admin/documents").json()["documents"][0]["id"]
    for path in [f"/data/uploads/{a_org}/{doc_id}.pdf", f"/uploads/{a_org}/{doc_id}.pdf",
                 f"/static/../data/uploads/{a_org}/{doc_id}.pdf", f"/api/documents/{doc_id}"]:
        assert anon.get(path).status_code == 404


def test_public_assistant_info_leaks_nothing_but_org_name(two_orgs):
    info = new_client().get(f"/api/assistant/{two_orgs['token_a']}").json()
    assert info == {"org_name": "Northbridge University"}


def test_invalid_and_regenerated_tokens_are_rejected(fake_llm):
    c = new_client()
    old = register(c, "Rotating Org")["token"]
    new = c.post("/api/admin/assistant/regenerate").json()["token"]
    assert new != old
    anon = new_client()
    assert anon.post("/api/ask", json={"token": old, "question": "hello?"}).status_code == 404
    assert anon.get(f"/a/{old}").status_code == 404
    assert anon.get(f"/a/{new}").status_code == 200
    assert anon.post("/api/ask", json={"token": "ast_doesnotexist", "question": "hi"}).status_code == 404
    assert anon.post("/api/ask", json={"token": "' OR 1=1 --", "question": "hi"}).status_code == 404


def test_dashboard_cannot_be_framed_but_assistant_can(two_orgs):
    anon = new_client()
    assert anon.get("/dashboard").headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors *" in anon.get(f"/a/{two_orgs['token_a']}").headers["Content-Security-Policy"]
