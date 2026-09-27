"""PDF ingestion: validation against the mandatory limits, text extraction (with OCR for
scanned pages) and chunking.

Chunks never cross page boundaries, so every retrieved passage maps to exactly one
(document, page) pair for source references.
"""
import io
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import settings
from .db import get_conn, utcnow
from .llm import LLMError, get_llm

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_SENTENCES = 1
MIN_TEXT_CHARS = 20            # a page with less extractable text than this is a candidate for OCR
OCR_MAX_SIDE_PX = 2000


class DocumentError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _clean_filename(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    name = re.sub(r"[^\w.\- ()]", "_", name)[:120]
    if not name.lower().endswith(".pdf"):
        raise DocumentError("Only PDF files are supported. Please upload a file ending in .pdf.", 415)
    return name


def _normalize(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def chunk_page(text: str) -> list[str]:
    """Pack sentences into ~CHUNK_TARGET_CHARS chunks with a one-sentence overlap."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?:])\s+|\n\n", text) if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        if current and sum(len(s) + 1 for s in current) + len(sentence) > CHUNK_TARGET_CHARS:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _has_images(page) -> bool:
    """True if the page draws at least one raster image (directly or inside a form XObject)."""
    def walk(resources, depth: int) -> bool:
        xobjects = (resources or {}).get("/XObject")
        if not xobjects or depth > 3:
            return False
        for ref in xobjects.get_object().values():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                return True
            if obj.get("/Subtype") == "/Form" and walk(obj.get("/Resources"), depth + 1):
                return True
        return False
    try:
        resources = page.get("/Resources")
        return walk(resources.get_object() if resources else None, 0)
    except Exception:
        return False


_pdfium_lock = threading.Lock()   # pdfium is not thread-safe: all rendering is serialized


def render_pages_png(data: bytes, indexes: list[int]) -> dict[int, bytes]:
    """Rasterize pages (grayscale PNG, longest side <= OCR_MAX_SIDE_PX) for OCR."""
    out = {}
    with _pdfium_lock:
        doc = pdfium.PdfDocument(data)
        try:
            for i in indexes:
                image = doc[i].render(scale=2).to_pil().convert("L")
                image.thumbnail((OCR_MAX_SIDE_PX, OCR_MAX_SIDE_PX))
                buf = io.BytesIO()
                image.save(buf, format="PNG", optimize=True)
                out[i] = buf.getvalue()
        finally:
            doc.close()
    return out


def ocr_pages(data: bytes, indexes: list[int]) -> dict[int, str]:
    """OCR image-only pages: render sequentially, then transcribe with the vision LLM in parallel."""
    try:
        images = render_pages_png(data, indexes)
    except Exception as exc:
        raise DocumentError("This PDF is scanned (its pages are images) and we couldn't render its pages "
                            "for OCR. Please try re-exporting it.", 422) from exc
    llm = get_llm()
    try:
        with ThreadPoolExecutor(max_workers=settings.ocr_concurrency) as pool:
            texts = pool.map(llm.ocr, [images[i] for i in indexes])
            return dict(zip(indexes, texts))
    except LLMError as exc:
        raise DocumentError(f"This PDF is scanned (its pages are images), so it needs OCR, which failed: "
                            f"{exc} Please try again, or upload a PDF with selectable text.", 422) from exc


def extract_pages(data: bytes) -> tuple[list[str], int]:
    """Return (normalized text per page, number of pages that were OCR'd)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("This PDF is password-protected. Please upload an unlocked copy.", 422)
        page_count = len(reader.pages)
    except DocumentError:
        raise
    except (PdfReadError, Exception) as exc:
        raise DocumentError(f"This file could not be read as a PDF ({type(exc).__name__}). "
                            "It may be corrupted.", 422) from exc

    if page_count == 0:
        raise DocumentError("This PDF has no pages.", 422)
    if page_count > settings.max_pages:
        raise DocumentError(f"This PDF has {page_count} pages. The maximum is "
                            f"{settings.max_pages} pages per document.", 413)
    try:
        raw = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentError("We couldn't extract text from this PDF. Please try re-exporting it.", 422) from exc

    # Scanned pages: no text layer but an image on the page -> OCR. Truly blank pages are skipped.
    scanned = [i for i, (text, page) in enumerate(zip(raw, reader.pages))
               if len(text.strip()) < MIN_TEXT_CHARS and _has_images(page)]
    if scanned:
        for i, text in ocr_pages(data, scanned).items():
            raw[i] = text
    return [_normalize(text) for text in strip_running_lines(raw)], len(scanned)


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_running_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers: lines (digits ignored, so 'Page 2 of 5' matches) that
    repeat on more than half of the pages. They add noise to every chunk and to retrieval."""
    if len(pages) < 2:
        return pages
    counts: dict[str, int] = {}
    for text in pages:
        for key in {_line_key(l) for l in text.splitlines() if l.strip()}:
            counts[key] = counts.get(key, 0) + 1
    running = {k for k, n in counts.items() if n > len(pages) / 2 and len(k) < 120}
    stripped = ["\n".join(l for l in text.splitlines() if _line_key(l) not in running) for text in pages]
    # Pages that are entirely "repeated" (e.g. identical pages) are real content, not headers.
    return stripped if any(t.strip() for t in stripped) else pages


def ingest_pdf(org_id: str, filename: str | None, data: bytes, replace_document_id: str | None = None) -> dict:
    name = _clean_filename(filename)
    if not data:
        raise DocumentError("The uploaded file is empty.", 400)
    if len(data) > settings.max_file_bytes:
        raise DocumentError(f"This file is {len(data) / 1_048_576:.1f} MB. The maximum size is "
                            f"{settings.max_file_bytes // 1_048_576} MB per PDF.", 413)
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise DocumentError("This file is not a valid PDF (missing PDF header).", 415)

    with get_conn() as conn:
        if replace_document_id:
            old = conn.execute("SELECT stored_path FROM documents WHERE id = ? AND org_id = ?",
                               (replace_document_id, org_id)).fetchone()
            if not old:
                raise DocumentError("The document to replace was not found.", 404)
        else:
            count = conn.execute("SELECT COUNT(*) FROM documents WHERE org_id = ?", (org_id,)).fetchone()[0]
            if count >= settings.max_documents:
                raise DocumentError(f"Your knowledge base already has {settings.max_documents} documents, "
                                    "which is the maximum. Delete or replace a document first.", 409)

    pages, ocr_count = extract_pages(data)
    page_chunks = [(page_no, chunk) for page_no, text in enumerate(pages, start=1)
                   for chunk in chunk_page(text)]
    if not page_chunks:
        raise DocumentError("No readable text was found in this PDF, even after OCR. It may be blank "
                            "or contain only pictures.", 422)

    doc_id = str(uuid.uuid4())
    org_dir = settings.upload_dir / org_id
    org_dir.mkdir(parents=True, exist_ok=True)
    stored_path = org_dir / f"{doc_id}.pdf"
    stored_path.write_bytes(data)

    try:
        with get_conn() as conn:
            if replace_document_id:
                old_path = conn.execute("SELECT stored_path FROM documents WHERE id = ? AND org_id = ?",
                                        (replace_document_id, org_id)).fetchone()
                conn.execute("DELETE FROM documents WHERE id = ? AND org_id = ?", (replace_document_id, org_id))
            conn.execute(
                """INSERT INTO documents (id, org_id, filename, stored_path, size_bytes, page_count,
                                          chunk_count, ocr_pages, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, org_id, name, str(stored_path), len(data), len(pages), len(page_chunks), ocr_count,
                 utcnow()),
            )
            conn.executemany(
                "INSERT INTO chunks (org_id, document_id, page, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
                [(org_id, doc_id, page, i, text) for i, (page, text) in enumerate(page_chunks)],
            )
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise DocumentError("The document could not be saved to the knowledge base. Please try again.", 500) from exc

    if replace_document_id and old_path:
        Path(old_path["stored_path"]).unlink(missing_ok=True)

    return get_document(org_id, doc_id)


def list_documents(org_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, filename, size_bytes, page_count, chunk_count, ocr_pages, uploaded_at
               FROM documents WHERE org_id = ? ORDER BY uploaded_at DESC""", (org_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(org_id: str, doc_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, filename, size_bytes, page_count, chunk_count, ocr_pages, uploaded_at
               FROM documents WHERE id = ? AND org_id = ?""", (doc_id, org_id)
        ).fetchone()
    if not row:
        raise DocumentError("Document not found.", 404)
    return dict(row)


def delete_document(org_id: str, doc_id: str) -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT stored_path FROM documents WHERE id = ? AND org_id = ?",
                           (doc_id, org_id)).fetchone()
        if not row:
            raise DocumentError("Document not found.", 404)
        conn.execute("DELETE FROM documents WHERE id = ? AND org_id = ?", (doc_id, org_id))
    Path(row["stored_path"]).unlink(missing_ok=True)
