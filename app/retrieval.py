"""Tenant-scoped lexical retrieval (Okapi BM25).

The index for an organization is built only from rows WHERE org_id = <that org>, so another
tenant's chunks are never candidates. Indexes are cached per org and automatically rebuilt
when that org's chunk set changes (chunk ids are AUTOINCREMENT, so COUNT+MAX(id) is a version).
"""
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

from .db import get_conn

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into onto over under as is are was
were be been being do does did doing have has had having i me my mine we our ours you your yours he
she it its they them their this that these those what which who whom whose when where why how can
could should would will shall may might must not no nor so than too very just also about any all each
both few more most other some such only own same there here up down out off again further once am
get got please tell know want need like s t don doesn isn aren m ve ll re d
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def stem(word: str) -> str:
    """Tiny suffix stripper; good enough to match plural/verb forms in policy documents."""
    if len(word) <= 3 or word[0].isdigit():
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                return word
            return word[: -len(suffix)] + repl
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


def _one_edit(a: str, b: str) -> bool:
    """True if a and b differ by exactly one insertion, deletion or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


@dataclass
class Passage:
    sid: str            # citation id shown to the LLM, e.g. "S1"
    chunk_id: int
    document_id: str
    filename: str
    page: int
    text: str
    score: float

    def reference(self) -> dict:
        return {"source_id": self.sid, "document": self.filename, "page": self.page,
                "snippet": self.text[:280] + ("…" if len(self.text) > 280 else "")}


@dataclass
class BM25Index:
    rows: list[dict]
    k1: float = 1.5
    b: float = 0.75
    doc_terms: list[Counter] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avgdl: float = 0.0

    _expansions: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.doc_terms = [Counter(tokenize(r["filename"].replace("_", " ") + " " + r["text"]))
                          for r in self.rows]
        for terms in self.doc_terms:
            self.df.update(terms.keys())
        total = sum(sum(t.values()) for t in self.doc_terms)
        self.avgdl = total / len(self.doc_terms) if self.doc_terms else 0.0

    @property
    def n(self) -> int:
        return len(self.rows)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def expand(self, term: str) -> dict[str, float]:
        """Corpus terms that match a query term, with a match weight.

        Exact match = 1.0. Close variants that actually occur in this org's documents are matched
        at a discount so lexical search tolerates abbreviations and typos:
          * affix variants: one term is a prefix/suffix of the other with <= 2 extra letters,
            e.g. gpa <-> cgpa, pdf <-> pdfs                                      -> 0.8
          * one-edit typos for terms of 5+ letters (scholarshp -> scholarship)   -> 0.7
        """
        if term in self._expansions:
            return self._expansions[term]
        matches = {term: 1.0} if term in self.df else {}
        if len(term) >= 3 and term.isalpha():
            for v in self.df:
                if v == term or not v.isalpha() or len(v) < 3 or abs(len(v) - len(term)) > 2:
                    continue
                short, long_ = sorted((term, v), key=len)
                if short != long_ and (long_.startswith(short) or long_.endswith(short)):
                    matches[v] = max(matches.get(v, 0), 0.8)
                elif len(term) >= 5 and _one_edit(term, v):
                    matches[v] = max(matches.get(v, 0), 0.7)
        self._expansions[term] = matches
        return matches

    def score(self, query_terms: list[str], i: int) -> float:
        terms, dl = self.doc_terms[i], sum(self.doc_terms[i].values())
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        s = 0.0
        for q in set(query_terms):
            # Best matching variant only, so one query term never counts twice.
            s += max((w * self.idf(v) * terms[v] * (self.k1 + 1) / (terms[v] + norm)
                      for v, w in self.expand(q).items() if terms.get(v)), default=0.0)
        return s

    def search(self, query: str, k: int) -> list[tuple[dict, float, Counter]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.rows[i], s, self.doc_terms[i]) for i, s in scored[:k]]


_cache: dict[str, tuple[tuple, BM25Index]] = {}
_lock = threading.Lock()


def get_index(org_id: str) -> BM25Index:
    with get_conn() as conn:
        version = tuple(conn.execute("SELECT COUNT(*), MAX(id) FROM chunks WHERE org_id = ?",
                                     (org_id,)).fetchone())
        with _lock:
            cached = _cache.get(org_id)
            if cached and cached[0] == version:
                return cached[1]
        rows = conn.execute(
            """SELECT c.id, c.document_id, c.page, c.text, d.filename
               FROM chunks c JOIN documents d ON d.id = c.document_id AND d.org_id = c.org_id
               WHERE c.org_id = ? ORDER BY c.id""", (org_id,)
        ).fetchall()
    index = BM25Index([dict(r) for r in rows])
    with _lock:
        _cache[org_id] = (version, index)
    return index


@dataclass
class RetrievalResult:
    passages: list[Passage]
    query_terms: list[str]
    index: BM25Index
    passage_terms: list[Counter]


def retrieve(org_id: str, query: str, k: int = 6) -> RetrievalResult:
    index = get_index(org_id)
    hits = index.search(query, k)
    passages = [Passage(sid=f"S{i}", chunk_id=row["id"], document_id=row["document_id"],
                        filename=row["filename"], page=row["page"], text=row["text"], score=round(score, 3))
                for i, (row, score, _) in enumerate(hits, start=1)]
    return RetrievalResult(passages, tokenize(query), index, [terms for _, _, terms in hits])
