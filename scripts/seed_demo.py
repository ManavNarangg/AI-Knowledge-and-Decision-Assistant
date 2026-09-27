"""Create the two demo organizations and load their sample PDFs.

    python -m scripts.seed_demo

Idempotent: existing demo accounts are reused and their documents re-uploaded.
"""
from pathlib import Path

from fastapi import HTTPException

from app import auth
from app.config import settings
from app.db import get_conn, init_db
from app.documents import delete_document, ingest_pdf, list_documents

ROOT = Path(__file__).resolve().parent.parent / "sample_data"
PASSWORD = "demo-pass-123"

DEMO_ORGS = [
    ("Northbridge University", "admin@northbridge.edu", ROOT / "northbridge_university"),
    ("Acme Logistics", "hr@acmelogistics.example", ROOT / "acme_logistics"),
]


def ensure_org(name: str, email: str) -> str:
    try:
        return auth.register_org(name, email, PASSWORD)["org_id"]
    except HTTPException as exc:
        if exc.status_code != 409:
            raise
        return auth.authenticate(email, PASSWORD)["org_id"]


def main() -> None:
    init_db()
    if not any(ROOT.glob("*/*.pdf")):
        raise SystemExit("No sample PDFs found. Run: python sample_data/generate_pdfs.py")
    for name, email, folder in DEMO_ORGS:
        org_id = ensure_org(name, email)
        for doc in list_documents(org_id):
            delete_document(org_id, doc["id"])
        for pdf in sorted(folder.glob("*.pdf")):
            doc = ingest_pdf(org_id, pdf.name, pdf.read_bytes())
            print(f"  {name}: {doc['filename']} - {doc['page_count']} pages, {doc['chunk_count']} chunks")
        with get_conn() as conn:
            token = conn.execute("SELECT assistant_token FROM organizations WHERE id = ?", (org_id,)).fetchone()[0]
        print(f"{name}\n  login:     {email} / {PASSWORD}\n  assistant: {settings.public_base_url}/a/{token}\n")


if __name__ == "__main__":
    main()
