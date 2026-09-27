const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let limits = { max_documents: 10, max_pages: 20, max_file_mb: 10 };
let assistant = null;
let embedMode = "script";
let replaceTarget = null;

async function api(url, options = {}) {
  const res = await fetch(url, options);
  if (res.status === 401) { location.href = "/"; throw new Error("Not logged in"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

function flash(message, kind = "success") {
  const el = $("flash");
  el.className = `alert ${kind}`;
  el.textContent = message;
  clearTimeout(flash.t);
  flash.t = setTimeout(() => el.classList.add("hidden"), 5000);
}

const fmtSize = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
const fmtDate = (s) => new Date(s).toLocaleString();
const STATUS_BADGE = {
  answered: ["ok", "Answered"], unable_to_answer: ["warn", "Not in KB"],
  needs_review: ["warn", "Needs review"], error: ["danger", "Error"],
};

async function loadMe() {
  const me = await api("/api/admin/me");
  limits = me.limits;
  $("org-name").textContent = me.org_name;
  $("user-email").textContent = me.email;
  $("lim-pages").textContent = limits.max_pages;
  $("lim-mb").textContent = limits.max_file_mb;
  document.title = `${me.org_name} — Dashboard`;
}

async function loadAssistant(data) {
  assistant = data || (await api("/api/admin/assistant"));
  $("assistant-url").value = assistant.assistant_url;
  $("open-assistant").href = assistant.assistant_url;
  renderEmbed();
}

function renderEmbed() {
  $("embed-code").textContent = embedMode === "script" ? assistant.script_embed : assistant.iframe_embed;
  $("tab-script").classList.toggle("active", embedMode === "script");
  $("tab-iframe").classList.toggle("active", embedMode === "iframe");
}

async function loadDocs() {
  const { documents } = await api("/api/admin/documents");
  const n = documents.length;
  $("limits").textContent = `${n} / ${limits.max_documents} documents`;
  $("doc-meter").style.width = `${(n / limits.max_documents) * 100}%`;
  $("docs").innerHTML = n ? documents.map((d) => `
    <tr>
      <td><strong>${esc(d.filename)}</strong></td>
      <td>${d.page_count}${d.ocr_pages ? ` <span class="badge neutral" title="${d.ocr_pages} scanned page(s) read with OCR">OCR ${d.ocr_pages}</span>` : ""}</td>
      <td>${fmtSize(d.size_bytes)}</td>
      <td>${d.chunk_count}</td>
      <td class="muted small">${fmtDate(d.uploaded_at)}</td>
      <td class="actions">
        <button class="btn small" data-replace="${esc(d.id)}">Replace</button>
        <button class="btn small danger" data-delete="${esc(d.id)}" data-name="${esc(d.filename)}">Delete</button>
      </td>
    </tr>`).join("") : `<tr><td colspan="6" class="muted">No documents yet — upload your first PDF above.</td></tr>`;
}

async function loadUsage() {
  const u = await api("/api/admin/usage");
  $("u-total").textContent = u.total;
  $("u-answered").textContent = u.answered;
  $("u-unable").textContent = u.unable_to_answer;
  $("u-review").textContent = u.needs_review + u.errors;
  $("recent").innerHTML = u.recent.length ? u.recent.map((q) => {
    const [cls, label] = STATUS_BADGE[q.status] || ["neutral", q.status];
    return `<tr><td>${esc(q.question)}</td><td class="muted small">${esc(q.question_type || "—")}</td>
      <td><span class="badge ${cls}">${label}</span></td></tr>`;
  }).join("") : `<tr><td colspan="3" class="muted">No questions yet.</td></tr>`;
}

async function uploadFiles(files, replaceId = null) {
  const box = $("upload-status");
  for (const file of files) {
    const line = document.createElement("div");
    line.className = "alert";
    line.textContent = `Processing ${file.name}… (scanned PDFs are read with OCR and can take a little longer)`;
    box.prepend(line);
    const form = new FormData();
    form.append("file", file);
    try {
      const url = replaceId ? `/api/admin/documents/${encodeURIComponent(replaceId)}/replace` : "/api/admin/documents";
      const doc = await api(url, { method: "POST", body: form });
      line.className = "alert success";
      line.textContent = `${replaceId ? "Replaced with" : "Added"} ${doc.filename}: ${doc.page_count} pages` +
        `${doc.ocr_pages ? ` (${doc.ocr_pages} scanned, read with OCR)` : ""}, ${doc.chunk_count} chunks indexed.`;
    } catch (e) {
      line.className = "alert error";
      line.textContent = `${file.name}: ${e.message}`;
    }
    setTimeout(() => line.remove(), 10000);
  }
  await loadDocs();
}

// ---- events
$("logout").onclick = async () => { await fetch("/api/auth/logout", { method: "POST" }); location.href = "/"; };
$("tab-script").onclick = () => { embedMode = "script"; renderEmbed(); };
$("tab-iframe").onclick = () => { embedMode = "iframe"; renderEmbed(); };
$("refresh-usage").onclick = () => loadUsage().catch((e) => flash(e.message, "error"));

document.querySelectorAll("[data-copy]").forEach((b) => b.onclick = () => copy($(b.dataset.copy).value));
$("copy-embed").onclick = () => copy($("embed-code").textContent);
async function copy(text) {
  try { await navigator.clipboard.writeText(text); flash("Copied to clipboard."); }
  catch { flash("Copy failed — select the text and copy manually.", "error"); }
}

$("regenerate").onclick = async () => {
  if (!confirm("Regenerate the assistant token? The current URL and embed code will stop working immediately.")) return;
  try { await loadAssistant(await api("/api/admin/assistant/regenerate", { method: "POST" })); flash("New assistant token generated. Update your embed code."); }
  catch (e) { flash(e.message, "error"); }
};

const dz = $("dropzone");
dz.onclick = () => $("file-input").click();
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("drag"); uploadFiles([...e.dataTransfer.files]); };
$("file-input").onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ""; };
$("replace-input").onchange = (e) => {
  if (e.target.files[0] && replaceTarget) uploadFiles([e.target.files[0]], replaceTarget);
  e.target.value = ""; replaceTarget = null;
};

$("docs").onclick = async (e) => {
  const rep = e.target.closest("[data-replace]");
  const del = e.target.closest("[data-delete]");
  if (rep) { replaceTarget = rep.dataset.replace; $("replace-input").click(); }
  if (del) {
    if (!confirm(`Delete ${del.dataset.name}? The assistant will no longer use it.`)) return;
    try { await api(`/api/admin/documents/${encodeURIComponent(del.dataset.delete)}`, { method: "DELETE" }); flash(`${del.dataset.name} deleted.`); await loadDocs(); }
    catch (err) { flash(err.message, "error"); }
  }
};

(async () => {
  try {
    await loadMe();
    await Promise.all([loadAssistant(), loadDocs(), loadUsage()]);
  } catch (e) { if (e.message !== "Not logged in") flash(e.message, "error"); }
})();
