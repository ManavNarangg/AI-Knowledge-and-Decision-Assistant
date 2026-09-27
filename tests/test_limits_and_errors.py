"""Upload limits and user-facing error handling (assignment sections 3 and 13)."""
from app.config import settings

from .conftest import SAMPLE, new_client, register, tiny_pdf, upload


def test_rejects_non_pdf():
    c = new_client(); register(c, "Limits Org 1")
    r = upload(c, SAMPLE / "invalid" / "not_really_a.pdf")
    assert r.status_code == 415 and "not a valid PDF" in r.json()["detail"]
    r = c.post("/api/admin/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and "Only PDF" in r.json()["detail"]


def test_rejects_too_many_pages():
    c = new_client(); register(c, "Limits Org 2")
    r = upload(c, SAMPLE / "invalid" / "too_many_pages_25.pdf")
    assert r.status_code == 413 and "25 pages" in r.json()["detail"]


def test_rejects_oversized_file():
    c = new_client(); register(c, "Limits Org 3")
    big = b"%PDF-1.4\n" + b"0" * settings.max_file_bytes
    r = c.post("/api/admin/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413 and "maximum size is 10 MB" in r.json()["detail"]


def test_rejects_pdf_without_text():
    c = new_client(); register(c, "Limits Org 4")
    r = upload(c, SAMPLE / "invalid" / "scanned_no_text.pdf")
    assert r.status_code == 422 and "No readable text" in r.json()["detail"]


def test_rejects_corrupt_pdf():
    c = new_client(); register(c, "Limits Org 5")
    r = c.post("/api/admin/documents", files={"file": ("broken.pdf", b"%PDF-1.7\n garbage", "application/pdf")})
    assert r.status_code == 422


def test_enforces_document_count_and_allows_replace():
    c = new_client(); register(c, "Limits Org 6")
    ids = []
    for i in range(settings.max_documents):
        r = c.post("/api/admin/documents", files={"file": (f"doc{i}.pdf", tiny_pdf(), "application/pdf")})
        assert r.status_code == 201
        ids.append(r.json()["id"])
    r = c.post("/api/admin/documents", files={"file": ("one_too_many.pdf", tiny_pdf(), "application/pdf")})
    assert r.status_code == 409 and "maximum" in r.json()["detail"]

    # Replacing is allowed at the limit and keeps the count at 10.
    r = c.post(f"/api/admin/documents/{ids[0]}/replace",
               files={"file": ("replacement.pdf", tiny_pdf(2), "application/pdf")})
    assert r.status_code == 200 and r.json()["page_count"] == 2
    docs = c.get("/api/admin/documents").json()["documents"]
    assert len(docs) == settings.max_documents and "replacement.pdf" in {d["filename"] for d in docs}

    assert c.delete(f"/api/admin/documents/{ids[1]}").status_code == 200
    assert len(c.get("/api/admin/documents").json()["documents"]) == settings.max_documents - 1


def test_ask_validation(two_orgs):
    anon = new_client()
    r = anon.post("/api/ask", json={"token": two_orgs["token_a"], "question": "   "})
    assert r.status_code == 400 and "type a question" in r.json()["detail"]
    r = anon.post("/api/ask", json={"token": two_orgs["token_a"], "question": "x" * 1001})
    assert r.status_code == 400
    assert anon.post("/api/ask", json={"question": "no token"}).status_code == 422


def test_registration_and_login_errors():
    c = new_client()
    assert c.post("/api/auth/register", json={"org_name": "X Org", "email": "bad", "password": "password123"}).status_code == 400
    assert c.post("/api/auth/register", json={"org_name": "X Org", "email": "a@b.co", "password": "short"}).status_code == 400
    assert c.post("/api/auth/register", json={"org_name": "X Org", "email": "dup@b.co", "password": "password123"}).status_code == 201
    assert new_client().post("/api/auth/register", json={"org_name": "Y", "email": "dup@b.co", "password": "password123"}).status_code in (400, 409)
    assert c.post("/api/auth/login", json={"email": "dup@b.co", "password": "wrong-pass"}).status_code == 401
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/admin/me").status_code == 401


def test_scanned_pdf_is_ocrd_and_searchable(fake_llm):
    c = new_client(); register(c, "Scanned Org")
    r = upload(c, SAMPLE / "scanned" / "Parking_Notice_scanned.pdf")
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["ocr_pages"] == 1 and doc["chunk_count"] >= 1
    assert [call["stage"] for call in fake_llm.calls] == ["ocr"] and fake_llm.calls[0]["bytes"] > 1000
    token = c.get("/api/admin/assistant").json()["token"]
    out = new_client().post("/api/ask", json={"token": token, "question": "How much does a parking permit cost?"}).json()
    assert out["status"] == "answered" and out["references"][0]["document"] == "Parking_Notice_scanned.pdf"


def test_scanned_pdf_ocr_failure_is_explained(fake_llm):
    fake_llm.fail_ocr = True
    c = new_client(); register(c, "Scanned Org 2")
    r = upload(c, SAMPLE / "scanned" / "Parking_Notice_scanned.pdf")
    assert r.status_code == 422 and "scanned" in r.json()["detail"] and "OCR" in r.json()["detail"]
    assert c.get("/api/admin/documents").json()["documents"] == []


def test_blank_pdf_is_not_sent_to_ocr(fake_llm):
    c = new_client(); register(c, "Blank Org")
    r = upload(c, SAMPLE / "invalid" / "scanned_no_text.pdf")
    assert r.status_code == 422 and fake_llm.calls == []
