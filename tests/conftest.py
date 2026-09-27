import os
import sys
import tempfile
from pathlib import Path

# Isolated data dir and no real LLM credentials for the whole test session (set before app import).
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="kda-test-")
os.environ["LLM_PROVIDER"] = "auto"
for _key in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
    os.environ[_key] = ""          # empty (not unset) so a developer's .env can't override it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from fpdf import FPDF  # noqa: E402

from app import llm as llm_module  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import Decomposition, GeneratedAnswer  # noqa: E402

settings.ask_rate_limit_per_minute = 10_000
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


class FakeLLM:
    """Scripted stand-in for Claude: records every call so tests can inspect what the LLM saw."""

    def __init__(self):
        self.decompositions: dict[str, Decomposition] = {}
        self.calls: list[dict] = []
        self.fail_decompose = False
        self.fail_generate = False
        self.fail_ocr = False
        self.grounded = True

    def decompose(self, question, passages, question_type):
        self.calls.append({"stage": "decompose", "question": question, "files": {p.filename for p in passages}})
        if self.fail_decompose:
            raise LLMError("decomposition unavailable")
        return self.decompositions.get(
            question, Decomposition(restated_question=question, facts=[], decisions=[], combine="none"))

    def ocr(self, png: bytes) -> str:
        self.calls.append({"stage": "ocr", "bytes": len(png)})
        if self.fail_ocr:
            raise LLMError("vision model unavailable.")
        return "Campus parking notice. Student parking permits cost $120 per semester. Overnight parking only in Lot C."

    def generate(self, question, passages, decision_context):
        self.calls.append({"stage": "generate", "question": question,
                           "files": {p.filename for p in passages}, "context": decision_context})
        if self.fail_generate:
            raise LLMError("The language model service is unavailable.")
        outcome = decision_context["overall_outcome"] if decision_context else "informational"
        return GeneratedAnswer(answer=f"Outcome: {outcome} [S1]", sources_used=["S1"], grounded=self.grounded)


@pytest.fixture
def fake_llm():
    fake = FakeLLM()
    llm_module.set_llm(fake)
    yield fake
    llm_module.set_llm(None)


@pytest.fixture(scope="session")
def app_client():
    with TestClient(app) as c:
        yield c


def new_client() -> TestClient:
    """A separate client = a separate cookie jar = a separate browser."""
    return TestClient(app)


_counter = [0]


def register(client: TestClient, org_name: str) -> dict:
    _counter[0] += 1
    email = f"user{_counter[0]}@{org_name.lower().replace(' ', '')}.test"
    r = client.post("/api/auth/register", json={"org_name": org_name, "email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    info = client.get("/api/admin/assistant").json()
    return {"email": email, "token": info["token"]}


def upload(client: TestClient, path: Path, name: str | None = None):
    return client.post("/api/admin/documents",
                       files={"file": (name or path.name, path.read_bytes(), "application/pdf")})


def tiny_pdf(pages: int = 1, text: str = "Tiny test document about widgets.") -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for _ in range(pages):
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


@pytest.fixture(scope="session")
def two_orgs(app_client):
    """Org A = Northbridge University docs, Org B = Acme Logistics docs, each in its own session."""
    if not any(SAMPLE.glob("*/*.pdf")):
        import runpy
        runpy.run_path(str(SAMPLE / "generate_pdfs.py"), run_name="__main__")
    a, b = new_client(), new_client()
    org_a, org_b = register(a, "Northbridge University"), register(b, "Acme Logistics")
    for pdf in (SAMPLE / "northbridge_university").glob("*.pdf"):
        assert upload(a, pdf).status_code == 201
    for pdf in (SAMPLE / "acme_logistics").glob("*.pdf"):
        assert upload(b, pdf).status_code == 201
    return {"a": a, "b": b, "token_a": org_a["token"], "token_b": org_b["token"]}
