import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth
from .config import settings
from .db import get_conn, init_db
from .documents import DocumentError, delete_document, ingest_pdf, list_documents
from .pipeline import answer_question

log = logging.getLogger("kda")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Knowledge & Decision Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

FRAMEABLE_PREFIXES = ("/a/", "/embed.js", "/static/")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    response.headers["X-Content-Type-Options"] = "nosniff"
    if path.startswith(FRAMEABLE_PREFIXES):
        # The visitor assistant must be embeddable on any customer site.
        response.headers["Content-Security-Policy"] = "frame-ancestors *"
        response.headers["Referrer-Policy"] = "no-referrer"
    else:
        # Dashboard and admin APIs can never be framed (clickjacking protection).
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
    return JSONResponse({"detail": f"Invalid {field}: {first.get('msg', 'bad value')}"}, status_code=422)


@app.exception_handler(DocumentError)
async def document_error(_: Request, exc: DocumentError):
    return JSONResponse({"detail": exc.message}, status_code=exc.status_code)


# --------------------------------------------------------------------------- pages

@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(settings.static_dir / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse(settings.static_dir / "dashboard.html")


@app.get("/a/{token}", include_in_schema=False)
def assistant_page(token: str):
    try:
        auth.org_for_assistant_token(token)
    except HTTPException:
        return HTMLResponse((settings.static_dir / "invalid.html").read_text(encoding="utf-8"), status_code=404)
    return FileResponse(settings.static_dir / "assistant.html")


@app.get("/embed.js", include_in_schema=False)
def embed_script():
    return FileResponse(settings.static_dir / "embed.js", media_type="application/javascript")


# --------------------------------------------------------------------------- auth

class RegisterBody(BaseModel):
    org_name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


def _set_session(response: Response, user_id: str, org_id: str) -> None:
    token = auth.create_session(user_id, org_id)
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True, samesite="strict",
                        secure=settings.cookie_secure, max_age=settings.session_ttl_hours * 3600)


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterBody, response: Response):
    ids = auth.register_org(body.org_name, body.email, body.password)
    _set_session(response, ids["user_id"], ids["org_id"])
    return {"ok": True}


@app.post("/api/auth/login")
def login(body: LoginBody, response: Response):
    ids = auth.authenticate(body.email, body.password)
    _set_session(response, ids["user_id"], ids["org_id"])
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth.destroy_session(request.cookies.get(auth.SESSION_COOKIE))
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


# --------------------------------------------------------------------------- admin (session required)

@app.get("/api/admin/me")
def me(admin: dict = Depends(auth.require_admin)):
    return {"org_name": admin["org_name"], "email": admin["email"],
            "limits": {"max_documents": settings.max_documents, "max_pages": settings.max_pages,
                       "max_file_mb": settings.max_file_bytes // 1_048_576}}


@app.get("/api/admin/documents")
def documents(admin: dict = Depends(auth.require_admin)):
    return {"documents": list_documents(admin["org_id"])}


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(settings.max_file_bytes + 1)   # never buffer more than limit + 1 byte
    await file.close()
    return data


@app.post("/api/admin/documents", status_code=201)
async def upload_document(file: UploadFile = File(...), admin: dict = Depends(auth.require_admin)):
    data = await _read_upload(file)
    # Ingestion (parsing + possible OCR calls) is blocking work: keep it off the event loop.
    return await asyncio.to_thread(ingest_pdf, admin["org_id"], file.filename, data)


@app.post("/api/admin/documents/{doc_id}/replace")
async def replace_document(doc_id: str, file: UploadFile = File(...), admin: dict = Depends(auth.require_admin)):
    data = await _read_upload(file)
    return await asyncio.to_thread(ingest_pdf, admin["org_id"], file.filename, data, doc_id)


@app.delete("/api/admin/documents/{doc_id}")
def remove_document(doc_id: str, admin: dict = Depends(auth.require_admin)):
    delete_document(admin["org_id"], doc_id)
    return {"ok": True}


def _assistant_payload(token: str) -> dict:
    url = f"{settings.public_base_url}/a/{token}"
    return {
        "token": token,
        "assistant_url": url,
        "iframe_embed": (f'<iframe src="{url}" title="Knowledge assistant" width="400" height="600" '
                         'style="border:1px solid #ddd;border-radius:12px" loading="lazy"></iframe>'),
        "script_embed": f'<script src="{settings.public_base_url}/embed.js" data-assistant-token="{token}" async></script>',
    }


@app.get("/api/admin/assistant")
def assistant_info(admin: dict = Depends(auth.require_admin)):
    with get_conn() as conn:
        token = conn.execute("SELECT assistant_token FROM organizations WHERE id = ?",
                             (admin["org_id"],)).fetchone()[0]
    return _assistant_payload(token)


@app.post("/api/admin/assistant/regenerate")
def regenerate(admin: dict = Depends(auth.require_admin)):
    return _assistant_payload(auth.regenerate_assistant_token(admin["org_id"]))


@app.get("/api/admin/usage")
def usage(admin: dict = Depends(auth.require_admin)):
    with get_conn() as conn:
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM questions WHERE org_id = ? GROUP BY status",
                                   (admin["org_id"],)).fetchall())
        recent = conn.execute(
            """SELECT question, status, question_type, latency_ms, created_at FROM questions
               WHERE org_id = ? ORDER BY created_at DESC LIMIT 15""", (admin["org_id"],)).fetchall()
    return {"total": sum(counts.values()), "answered": counts.get("answered", 0),
            "unable_to_answer": counts.get("unable_to_answer", 0),
            "needs_review": counts.get("needs_review", 0), "errors": counts.get("error", 0),
            "recent": [dict(r) for r in recent]}


# --------------------------------------------------------------------------- visitor (assistant token)

class AskBody(BaseModel):
    token: str
    question: str


_ask_log: dict[str, deque] = defaultdict(deque)


def _rate_limited(key: str) -> bool:
    now, window = time.monotonic(), _ask_log[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.ask_rate_limit_per_minute:
        return True
    window.append(now)
    return False


@app.get("/api/assistant/{token}")
def assistant_public_info(token: str):
    org = auth.org_for_assistant_token(token)
    return {"org_name": org["org_name"]}


@app.post("/api/ask")
async def ask(body: AskBody, request: Request):
    org = auth.org_for_assistant_token(body.token.strip())
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > settings.max_question_chars:
        raise HTTPException(400, f"Questions are limited to {settings.max_question_chars} characters.")
    client = request.client.host if request.client else "unknown"
    if _rate_limited(f"{body.token}:{client}"):
        raise HTTPException(429, "You're asking questions too quickly. Please wait a moment.")
    try:
        return await answer_question(org["org_id"], org["org_name"], question)
    except Exception:
        log.exception("ask pipeline failed")
        raise HTTPException(500, "Something went wrong while answering. Please try again.")
