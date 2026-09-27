"""Organization accounts, admin sessions and assistant tokens.

Two credential types exist and never overlap:
  * Admin session  - random cookie value, stored only as a SHA-256 hash, required for /api/admin/*.
  * Assistant token - public, revocable identifier embedded in the assistant URL. It is accepted
    only by the visitor endpoints (/api/ask, /api/assistant/*) and grants nothing else.
"""
import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from .config import settings
from .db import get_conn, utcnow

SESSION_COOKIE = "kda_session"
PBKDF2_ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


def new_assistant_token() -> str:
    # 192 bits of randomness; carries no information about the org, its users or documents.
    return "ast_" + secrets.token_urlsafe(24)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def register_org(org_name: str, email: str, password: str) -> dict:
    org_name, email = org_name.strip(), email.strip().lower()
    if not (2 <= len(org_name) <= 100):
        raise HTTPException(400, "Organization name must be between 2 and 100 characters.")
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    org_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
    with get_conn() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists.")
        conn.execute(
            "INSERT INTO organizations (id, name, assistant_token, created_at) VALUES (?, ?, ?, ?)",
            (org_id, org_name, new_assistant_token(), utcnow()),
        )
        conn.execute(
            "INSERT INTO users (id, org_id, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, org_id, email, hash_password(password), utcnow()),
        )
    return {"user_id": user_id, "org_id": org_id}


def authenticate(email: str, password: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, org_id, password_hash FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")
    return {"user_id": row["id"], "org_id": row["org_id"]}


def create_session(user_id: str, org_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_ttl_hours)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, org_id, expires_at) VALUES (?, ?, ?, ?)",
            (_hash_token(token), user_id, org_id, expires.isoformat(timespec="seconds")),
        )
    return token


def destroy_session(token: str | None) -> None:
    if token:
        with get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


def require_admin(request: Request) -> dict:
    """FastAPI dependency for every administrative endpoint.

    Only the session cookie is consulted. Assistant tokens (in headers, query or body) are
    deliberately ignored, so an assistant token can never reach the dashboard APIs.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(401, "Please log in to continue.")
    with get_conn() as conn:
        row = conn.execute(
            """SELECT s.user_id, s.org_id, s.expires_at, u.email, o.name AS org_name
               FROM sessions s
               JOIN users u ON u.id = s.user_id
               JOIN organizations o ON o.id = s.org_id
               WHERE s.token_hash = ?""",
            (_hash_token(token),),
        ).fetchone()
        if not row:
            raise HTTPException(401, "Your session has expired. Please log in again.")
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
            raise HTTPException(401, "Your session has expired. Please log in again.")
    return dict(row)


def org_for_assistant_token(token: str) -> dict:
    """Server-side validation of an assistant token. Returns the owning organization."""
    if not token or not token.startswith("ast_") or len(token) > 100:
        raise HTTPException(404, "This assistant link is invalid or has been revoked.")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name FROM organizations WHERE assistant_token = ?", (token,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "This assistant link is invalid or has been revoked.")
    return {"org_id": row["id"], "org_name": row["name"]}


def regenerate_assistant_token(org_id: str) -> str:
    token = new_assistant_token()
    with get_conn() as conn:
        conn.execute("UPDATE organizations SET assistant_token = ? WHERE id = ?", (token, org_id))
    return token
