const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const token = decodeURIComponent(location.pathname.split("/a/")[1] || "").replace(/\/$/, "");
const log = $("log");

const STATUS = {
  unable_to_answer: ["warn", "Not found in knowledge base"],
  needs_review: ["warn", "Partially automated — please verify"],
  error: ["danger", "Answer service unavailable"],
};
const COMPONENT_BADGE = { "LLM": "llm", "System 1": "sys1", "App logic": "app" };

fetch(`/api/assistant/${encodeURIComponent(token)}`).then((r) => r.json()).then((d) => {
  if (!d.org_name) return;
  $("org-name").textContent = d.org_name;
  $("org-name-2").textContent = d.org_name;
  $("org-initial").textContent = d.org_name.trim()[0].toUpperCase();
  document.title = `${d.org_name} Assistant`;
}).catch(() => {});

function renderMarkdown(text, refs) {
  const byId = Object.fromEntries(refs.flatMap((r) => (r.source_ids || [r.source_id]).map((id) => [id, r])));
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>")
    .replace(/\[(S\d+)\]/g, (_, id) => {
      const r = byId[id];
      const n = refs.indexOf(r) + 1;
      return r ? `<span class="cite" title="${esc(r.document)} — page ${r.page}">${n}</span>` : "";
    });
  const out = [];
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    if (bullet) { list = list || []; list.push(`<li>${inline(bullet[1])}</li>`); continue; }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line.startsWith(">")) out.push(`<blockquote>${inline(line.replace(/^>\s*/, ""))}</blockquote>`);
    else if (line) out.push(`<p>${inline(line.replace(/^#+\s*/, ""))}</p>`);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

function renderTrace(t) {
  if (!t || !t.stages) return "";
  const stages = t.stages.map((s) => {
    const extra = Object.entries(s).filter(([k]) => !["stage", "component", "ms"].includes(k))
      .map(([k, v]) => `${k}: ${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(", ");
    return `<div class="stage"><span class="badge ${COMPONENT_BADGE[s.component] || "neutral"}">${esc(s.component)}</span>
      <span>${esc(s.stage)}</span><span class="ms">${s.ms} ms</span>${extra ? `<span class="muted">${extra}</span>` : ""}</div>`;
  }).join("");

  const c = t.classification;
  const probs = c ? Object.entries(c.probabilities).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(" · ") : "";

  const facts = t.decomposition?.facts?.length
    ? `<div><strong>Facts extracted:</strong> ${t.decomposition.facts.map((f) => `<code>${esc(f.name)}=${esc(f.value)}</code>`).join(" ")}</div>` : "";

  const decisions = (t.decisions || []).map((d) => {
    const outcome = d.error ? `<span class="tri-false">error</span>`
      : d.type === "score" ? `<strong>${d.outcome}</strong> (${esc(d.band)})`
      : `<span class="tri-${esc(d.outcome)}">${esc(d.outcome)}</span>`;
    const conds = (d.conditions || []).map((x) =>
      `<div class="cond"><span class="tri-${x.result}">${x.result.toUpperCase()}</span> ${esc(x.description)} — ` +
      `${esc(x.fact)}=${esc(x.actual ?? "?")} ${esc(x.operator)} ${esc(x.expected)} [${esc(x.source)}]</div>`).join("");
    return `<div class="decision"><div><span class="badge sys1">${esc(d.type)}</span> <strong>${esc(d.id)}</strong> ${esc(d.question)} → ${outcome}</div>
      ${conds}${d.error ? `<div class="cond tri-false">${esc(d.error)}</div>` : ""}</div>`;
  }).join("");

  const combined = t.combined ? `<div><span class="badge app">App logic</span> combine <code>${esc(t.decomposition?.combine)}</code> → overall <strong>${esc(t.combined.overall)}</strong>${
    t.combined.missing_facts.length ? ` · missing: ${t.combined.missing_facts.map((m) => `<code>${esc(m)}</code>`).join(" ")}` : ""}</div>` : "";

  return `<details class="trace"><summary>How this answer was produced (${t.llm_calls} LLM call${t.llm_calls === 1 ? "" : "s"})</summary>
    <div class="trace-body">
      ${stages}
      ${c ? `<div><strong>Question type:</strong> ${esc(c.label)} <span class="muted">(${probs})</span></div>` : ""}
      ${t.evidence ? `<div><strong>Evidence score:</strong> ${t.evidence.score} (${esc(t.evidence.verdict)})</div>` : ""}
      ${t.memory && t.memory.resumed ? `<div><strong>Memory:</strong> continuing conversation (${t.memory.history_messages} earlier messages)${
        t.memory.remembered_facts.length ? ` · remembered facts: ${t.memory.remembered_facts.map((f) => `<code>${esc(f)}</code>`).join(" ")}` : ""}${
        t.memory.search_used_conversation ? " · follow-up searched together with earlier questions" : ""}${
        t.memory.awaiting_facts ? ` · was waiting for: ${t.memory.awaiting_facts.missing_facts.map((f) => `<code>${esc(f)}</code>`).join(" ")}` : ""}</div>` : ""}
      ${t.route ? `<div><strong>Route:</strong> ${t.route.decomposition ? "decompose into structured decisions" : "answer directly"} — ${esc(t.route.reason)}</div>` : ""}
      ${facts}${decisions}${combined}
    </div></details>`;
}

function addMessage(role, html) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

async function ask(question) {
  addMessage("user", esc(question));
  const pending = addMessage("bot", `<div class="typing"><span></span><span></span><span></span></div>`);
  $("send").disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, question, conversation_id: conversationId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Something went wrong. Please try again.");
    setConversation(data.conversation_id);
    pending.innerHTML = renderAnswer(data.answer, data.status, data.references || []) + renderTrace(data.trace);
  } catch (e) {
    pending.innerHTML = `<span class="badge danger">Error</span><p>${esc(e.message)}</p>`;
  } finally {
    $("send").disabled = false;
    log.scrollTop = log.scrollHeight;
  }
}

// ---- conversation memory: the server keeps the conversation; the browser only remembers its id.
function renderAnswer(answer, statusKey, refs) {
  const status = STATUS[statusKey];
  return (status ? `<div class="status-line"><span class="badge ${status[0]}">${status[1]}</span></div>` : "") +
    renderMarkdown(answer, refs) +
    (refs.length ? `<div class="refs"><div class="muted">Sources</div>${refs.map((r, i) =>
      `<div class="ref" title="${esc(r.snippet)}"><b>${i + 1}</b> ${esc(r.document)} — Page ${r.page}</div>`).join("")}</div>` : "");
}

const CONV_KEY = `kda_conversation_${token}`;
let conversationId = null;
try { conversationId = localStorage.getItem(CONV_KEY); } catch { /* storage unavailable: memory lasts for this page only */ }

function setConversation(id) {
  conversationId = id;
  try { id ? localStorage.setItem(CONV_KEY, id) : localStorage.removeItem(CONV_KEY); } catch { /* ignore */ }
}

async function restoreConversation() {
  if (!conversationId) return;
  try {
    const res = await fetch(`/api/assistant/${encodeURIComponent(token)}/conversations/${encodeURIComponent(conversationId)}`);
    if (!res.ok) { setConversation(null); return; }          // expired or unknown: start fresh
    const { messages } = await res.json();
    for (const m of messages) {
      if (m.role === "user") addMessage("user", esc(m.content));
      else addMessage("bot", renderAnswer(m.content, m.status, m.references || []));
    }
    if (messages.length) addMessage("bot", `<p class="muted small">Continuing your previous conversation.</p>`);
  } catch { /* offline: just start fresh */ }
}

$("new-chat").onclick = () => {
  setConversation(null);
  [...log.querySelectorAll(".msg")].slice(1).forEach((el) => el.remove());   // keep the welcome message
  $("q").focus();
};
restoreConversation();

$("form").onsubmit = (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  $("q").value = "";
  ask(q);
};
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); }
});
