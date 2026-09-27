"""Conversation memory for visitors.

Two kinds of memory are kept per conversation:
  * Transcript  - the recent user/assistant messages, so follow-ups ("what about Gold tier?")
                  can be resolved by retrieval and by the LLM stages.
  * Fact memory - the visitor facts extracted so far (gpa=3.72, year_of_study=2, ...) plus any
                  decision still waiting for facts. The System 1 evaluators reuse these facts
                  deterministically, so "I need your GPA" -> "3.9" completes the original check.

Tenant isolation: a conversation belongs to exactly one organization. A conversation id that
is unknown, expired, or belongs to another organization is ignored and a new one is started.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .db import get_conn, utcnow

CONVERSATION_TTL = timedelta(days=7)      # idle conversations are not resumed after this
HISTORY_MESSAGES = 6                       # recent messages shown to the LLM stages
MAX_MESSAGE_CHARS = 700                    # per message, when shown to the LLM


@dataclass
class Conversation:
    id: str
    org_id: str
    is_new: bool
    history: list[dict] = field(default_factory=list)   # [{"role": "user"|"assistant", "content": str}]
    facts: dict[str, dict] = field(default_factory=dict)  # name -> {"name", "value", "kind"}
    pending: dict | None = None          # {"question": str, "missing_facts": [...]} awaiting facts

    def recent_user_questions(self, n: int = 2) -> list[str]:
        return [m["content"] for m in self.history if m["role"] == "user"][-n:]


def _expired(updated_at: str) -> bool:
    return datetime.fromisoformat(updated_at) < datetime.now(timezone.utc) - CONVERSATION_TTL


def load_or_start(org_id: str, conversation_id: str | None) -> Conversation:
    """Resume the visitor's conversation if it is valid for this org, otherwise start a new one."""
    if conversation_id and len(conversation_id) <= 64:
        with get_conn() as conn:
            row = conn.execute("SELECT id, state, updated_at FROM conversations WHERE id = ? AND org_id = ?",
                               (conversation_id, org_id)).fetchone()
            if row and not _expired(row["updated_at"]):
                msgs = conn.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? AND org_id = ? "
                    "ORDER BY id DESC LIMIT ?", (row["id"], org_id, HISTORY_MESSAGES)).fetchall()
                state = json.loads(row["state"] or "{}")
                return Conversation(row["id"], org_id, False,
                                    history=[{"role": m["role"], "content": m["content"]} for m in reversed(msgs)],
                                    facts=state.get("facts", {}), pending=state.get("pending"))
    return Conversation(uuid.uuid4().hex, org_id, True)


def save_turn(conv: Conversation, question: str, answer: str, status: str, references: list[dict]) -> None:
    now = utcnow()
    state = json.dumps({"facts": conv.facts, "pending": conv.pending})
    with get_conn() as conn:
        if conv.is_new:
            conn.execute("INSERT INTO conversations (id, org_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                         (conv.id, conv.org_id, state, now, now))
        else:
            conn.execute("UPDATE conversations SET state = ?, updated_at = ? WHERE id = ? AND org_id = ?",
                         (state, now, conv.id, conv.org_id))
        conn.executemany(
            "INSERT INTO messages (conversation_id, org_id, role, content, status, references_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(conv.id, conv.org_id, "user", question[:2000], None, None, now),
             (conv.id, conv.org_id, "assistant", answer, status, json.dumps(references), now)])
    conv.is_new = False


def transcript(org_id: str, conversation_id: str, limit: int = 50) -> list[dict] | None:
    """Messages of a conversation for re-display in the chat UI (None if not found for this org)."""
    with get_conn() as conn:
        row = conn.execute("SELECT updated_at FROM conversations WHERE id = ? AND org_id = ?",
                           (conversation_id, org_id)).fetchone()
        if not row or _expired(row["updated_at"]):
            return None
        rows = conn.execute(
            "SELECT role, content, status, references_json FROM messages WHERE conversation_id = ? AND org_id = ? "
            "ORDER BY id DESC LIMIT ?", (conversation_id, org_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"], "status": r["status"],
             "references": json.loads(r["references_json"]) if r["references_json"] else []}
            for r in reversed(rows)]


def format_history(history: list[dict]) -> str:
    """Recent turns for the LLM prompt (each message truncated)."""
    lines = []
    for m in history[-HISTORY_MESSAGES:]:
        text = " ".join(m["content"].split())
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "…"
        lines.append(f"{'Visitor' if m['role'] == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)
