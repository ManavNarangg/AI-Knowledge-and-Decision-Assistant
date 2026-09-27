# How It Works: plain English + the actual code

This guide explains how the assistant answers a question. Each step has two parts:

- **In plain words:** what the step does and why, with an example.
- **In the code:** the real code (trimmed a little), a link to the exact lines, and a walk-through using real numbers from the demo data.

For setup instructions, see [README.md](README.md).

---

## The idea in one sentence

> **The AI reads and writes. Plain code decides.**

Think of an office with two workers:

- **The clerk (the AI, e.g. Gemini)** is good with language. It reads a messy question and a rulebook, fills in a neat **form**, and later writes a friendly reply. It is *not* trusted to make the decision, because language models can get checks like "is 3.72 ≥ 3.5?" wrong or make things up.
- **The calculator (plain Python)** can't understand English, but it never gets a rule check wrong. It takes the clerk's form and computes the answer.

**Clerk fills in a form → calculator checks it → clerk writes the reply.**

The assignment calls the calculator-style parts **"System 1 models"**: fast, predictable, no AI.

---

## The call chain at a glance

Every question follows this chain of function calls. The numbers ①–⑨ match the steps below.

```
POST /api/ask ─► main.ask()                                   app/main.py:234
                  └─ auth.org_for_assistant_token()        ①   app/auth.py:126
                  └─ pipeline.answer_question()                app/pipeline.py:109
                       ├─ small_talk_reply()                    app/pipeline.py:45
                       ├─ asyncio.gather(                       app/pipeline.py:130
                       │     retrieval.retrieve()           ②   app/retrieval.py:170
                       │     classifier.predict() )         ③   app/system1/classifier.py:126
                       ├─ evidence.assess()                 ④   app/system1/evidence.py:37
                       ├─ needs_decisions = ...             ⑤   app/pipeline.py:158
                       ├─ llm.decompose()                   ⑥   app/llm.py:104
                       ├─ decisions.run_decisions()         ⑦   app/system1/decisions.py:224
                       ├─ combiner.combine()                ⑧   app/combiner.py:23
                       ├─ llm.generate()                    ⑨   app/llm.py:112
                       └─ _references() + _log()                app/pipeline.py:86, :100
```

---

## Part 1: Before anyone asks anything (uploading a PDF)

### In plain words

```
PDF ──► check it ──► text of each page (OCR if scanned) ──► cut into pieces ──► save with org + page
```

1. **Check it:** real PDF, ≤ 10 MB, ≤ 20 pages, and fewer than 10 documents per organization.
2. **Get each page's text.** Scanned pages (pictures, no text) are "read" by the AI; this is **OCR**. Repeated header/footer lines like "Page 2 of 5" are removed.
3. **Cut each page into ~900-character pieces ("chunks").** A piece never crosses pages, so it always knows its page number. That is what makes "Page 5" citations possible.
4. **Save** each piece with its organization and page.

### In the code

**`ingest_pdf()`** ([documents.py:195-223](app/documents.py#L195-L223)), the main steps:

```python
pages, ocr_count = extract_pages(data)                        # text per page (+ OCR)
page_chunks = [(page_no, chunk) for page_no, text in enumerate(pages, start=1)
               for chunk in chunk_page(text)]                  # [(1, "..."), (1, "..."), (2, "...")]
...
conn.execute("INSERT INTO documents (...) VALUES (...)", (doc_id, org_id, name, ...))
conn.executemany(
    "INSERT INTO chunks (org_id, document_id, page, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
    [(org_id, doc_id, page, i, text) for i, (page, text) in enumerate(page_chunks)])
```

- Each chunk row stores **`org_id`** (whose it is) and **`page`** (where it came from). Everything later relies on these two columns.

**Detecting scanned pages** in `extract_pages()` ([documents.py:141-150](app/documents.py#L141-L150)):

```python
raw = [page.extract_text() or "" for page in reader.pages]
scanned = [i for i, (text, page) in enumerate(zip(raw, reader.pages))
           if len(text.strip()) < MIN_TEXT_CHARS and _has_images(page)]   # < 20 chars AND has a picture
if scanned:
    for i, text in ocr_pages(data, scanned).items():                        # render page → AI reads it
        raw[i] = text
```

- A page is sent to OCR only if it has almost no text **and** contains an image. Blank pages are skipped.

**`chunk_page()`** ([documents.py:51-62](app/documents.py#L51-L62)) packs sentences into chunks:

```python
sentences = [s.strip() for s in re.split(r"(?<=[.!?:])\s+|\n\n", text) if s.strip()]
chunks, current = [], []
for sentence in sentences:
    if current and sum(len(s) + 1 for s in current) + len(sentence) > CHUNK_TARGET_CHARS:  # 900
        chunks.append(" ".join(current))
        current = current[-1:]          # start the next chunk with the last sentence (overlap)
    current.append(sentence)
if current:
    chunks.append(" ".join(current))
```

- It splits the page into sentences, then adds sentences to the current chunk until the next one would push it past 900 characters.
- When a chunk is closed, the next one starts by repeating its last sentence. That way a rule sitting on the boundary between two chunks is never lost.

---

## Part 2: Answering a question (the core algorithm)

Our example question, followed all the way through:

> **"I'm a second-year full-time student with a 3.72 GPA and no misconduct record. I already have a sports scholarship worth $4,000 a year. Am I eligible for the Merit Scholarship, and which tier would I get?"**

```
 ①  Which organization?                 code
 ②  Find relevant pieces of text        code ─┐ run at the
 ③  What kind of question?              code ─┘ same time
 ④  Do the documents cover it?          code ── no ──► "Not in the knowledge base"
 ⑤  Needs a yes/no decision?            code ── no ──► skip to ⑨
 ⑥  Question + rules → form             AI
 ⑦  Check every rule                    code (all at the same time)
 ⑧  Combine the results                 code
 ⑨  Write the answer with sources       AI
```

---

### ① Which organization is this?

**In plain words.** The chat sends the question plus the **assistant token** from its link. The server looks up which organization owns that token. From then on, only that organization's documents are used.

**In the code.** The entry point `main.ask()` ([main.py:234-248](app/main.py#L234-L248)):

```python
org = auth.org_for_assistant_token(body.token.strip())       # token → {org_id, org_name}, or 404
question = body.question.strip()
if not question:
    raise HTTPException(400, "Please type a question.")
if len(question) > settings.max_question_chars:              # 1000
    raise HTTPException(400, ...)
if _rate_limited(f"{body.token}:{client}"):                  # max 30 per minute
    raise HTTPException(429, ...)
return await answer_question(org["org_id"], org["org_name"], question)
```

`org_for_assistant_token()` ([auth.py:126-136](app/auth.py#L126-L136)) is a single database lookup:

```python
row = conn.execute("SELECT id, name FROM organizations WHERE assistant_token = ?", (token,)).fetchone()
if not row:
    raise HTTPException(404, "This assistant link is invalid or has been revoked.")
return {"org_id": row["id"], "org_name": row["name"]}
```

- The `org_id` found here is passed into everything that follows. Nothing else can pick which organization is used.

---

### ② Find the most relevant pieces of text (BM25 search)

**In plain words.**

1. Turn the question into keywords: lowercase, drop filler words, trim word endings.
2. Score every piece of the organization's text:
   - **Rare keywords are worth more.** A word found in only a few pieces is a strong clue.
   - **Repeats help, with diminishing returns.**
   - **Close spellings count at a discount:** "gpa" ≈ "cgpa" at 80%, and a one-letter typo at 70%.
3. Keep the top 6 pieces and label them S1–S6.

**In the code.**

**Step 2a: keywords.** `tokenize()` ([retrieval.py:39-40](app/retrieval.py#L39-L40)):

```python
def tokenize(text):
    return [stem(w) for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]
```

- `TOKEN_RE` finds words and numbers (it keeps `3.5` as one token).
- `STOPWORDS` drops filler words like *what, do, I, the, for*.
- `stem()` ([retrieval.py:27](app/retrieval.py#L27)) trims endings: `scholarships → scholarship`, `rules → rule`.

Example: `"What GPA do I need for the Merit Scholarship?"` → `['gpa', 'merit', 'scholarship']`.

**Step 2b: load only this organization's pieces.** `get_index()` ([retrieval.py:143-159](app/retrieval.py#L143-L159)):

```python
version = conn.execute("SELECT COUNT(*), MAX(id) FROM chunks WHERE org_id = ?", (org_id,)).fetchone()
if cached and cached[0] == version:           # nothing changed since last time → reuse
    return cached[1]
rows = conn.execute("SELECT ... FROM chunks c JOIN documents d ... WHERE c.org_id = ?", (org_id,))
index = BM25Index(rows)
```

- **This `WHERE c.org_id = ?` is where tenant isolation happens.** Other organizations' text is never loaded.
- The index is cached. Any upload or delete changes `COUNT`/`MAX(id)`, so the cache rebuilds automatically.

**Step 2c: rarity weight (IDF).** `idf()` ([retrieval.py:93-95](app/retrieval.py#L93-L95)):

```python
def idf(self, term):
    df = self.df.get(term, 0)                              # how many pieces contain the word
    return math.log(1 + (self.n - df + 0.5) / (df + 0.5))  # n = total pieces
```

Real numbers for Northbridge (12 pieces in total):

| Word | Pieces containing it (`df`) | Weight (`idf`) |
|---|---|---|
| gpa | 6 | 0.693 |
| merit | 7 | 0.550 |
| scholarship | 8 | 0.425 (common, so a weak clue) |
| cafeteria | 0 | 3.258 (never seen, so the maximum weight) |

**Step 2d: score one piece.** `score()` ([retrieval.py:121-129](app/retrieval.py#L121-L129)):

```python
terms, dl = self.doc_terms[i], sum(self.doc_terms[i].values())   # word counts, piece length
norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)         # k1 = 1.5, b = 0.75
s = 0.0
for q in set(query_terms):
    s += max((w * self.idf(v) * terms[v] * (self.k1 + 1) / (terms[v] + norm)
              for v, w in self.expand(q).items() if terms.get(v)), default=0.0)
return s
```

Per keyword, this computes `weight × rarity × tf·(k1+1) / (tf + norm)`:

- `tf` = how many times the word appears in this piece.
- `tf·(k1+1)/(tf+norm)` grows with `tf` but levels off. That is the "diminishing returns" part.
- `norm` is larger for long pieces, so long pieces don't win just by being long.
- `expand(q)` supplies the spelling variants and their weights `w`. `max(...)` takes the best variant, so a keyword never counts twice.

Real calculation for the top result (Scholarship_Rules page 6, 56 words long, `norm = 1.401`):

| Keyword | tf | idf | Contribution |
|---|---|---|---|
| scholarship | 6 | 0.425 | 0.425 × 6 × 2.5 / (6 + 1.401) = **0.861** |
| gpa | 3 | 0.693 | 0.693 × 3 × 2.5 / (3 + 1.401) = **1.181** |
| merit | 1 | 0.550 | 0.550 × 1 × 2.5 / (1 + 1.401) = **0.573** |
| **Total** | | | **2.615** → ranked S1 |

**Step 2e: spelling variants.** `expand()` ([retrieval.py:97-119](app/retrieval.py#L97-L119)):

```python
matches = {term: 1.0} if term in self.df else {}          # exact match
for v in self.df:                                          # every word in this org's documents
    if abs(len(v) - len(term)) > 2: continue
    short, long_ = sorted((term, v), key=len)
    if long_.startswith(short) or long_.endswith(short):
        matches[v] = 0.8                                   # gpa ↔ cgpa
    elif len(term) >= 5 and _one_edit(term, v):
        matches[v] = 0.7                                   # scholarshp → scholarship
```

**Step 2f: keep the top pieces.** `search()` + `retrieve()` ([retrieval.py:131-176](app/retrieval.py#L131-L176)) score every piece, sort them, and keep the top `k`, labelled `S1, S2, …`.

---

### ③ What kind of question is it? (Naive Bayes classifier)

**In plain words.** A small model was trained on about 70 example questions labelled *factual, policy, procedural, decision* or *comparison*. It learned which words go with each type. "am I", "eligible" and numbers point to **decision**; "how do I" points to **procedural**. For a new question it adds up the clues and picks the most likely type. This step runs at the same time as ②.

**In the code.**

**Features** ([classifier.py:101-103](app/system1/classifier.py#L101-L103)):

```python
words = ["<num>" if w[0].isdigit() else w for w in TOKEN_RE.findall(text.lower())]
return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]      # single words + word pairs
```

`"Am I eligible for the merit scholarship with a 3.72 GPA?"` becomes `am, i, eligible, …, <num>, gpa, am_i, i_eligible, eligible_for, …`.

- Filler words like "am" and "I" are **kept** here, because they are the clues.
- Every number becomes `<num>`, so "3.72" and "40000" count as the same clue.

**Training** happens once, at startup ([classifier.py:114-124](app/system1/classifier.py#L114-L124)):

```python
for label, examples in data.items():
    self.priors[label] = math.log(len(examples) / total)     # how common each type is
    for ex in examples:
        self.counts[label].update(features(ex))              # how often each clue appears per type
```

**Prediction** ([classifier.py:126-140](app/system1/classifier.py#L126-L140)):

```python
log_scores = {
    label: self.priors[label] + sum(
        math.log((self.counts[label][f] + self.alpha) / (self.totals[label] + self.alpha * v))
        for f in feats)
    for label in self.priors}
# softmax → probabilities; the highest one is the label
```

- For each type, it adds up `log(how often this clue appeared in that type's examples)` over the question's clues.
- `alpha = 0.5` ("smoothing") stops a clue that was never seen for a type from making that type impossible.
- Using logs turns a product of small probabilities into a sum, which avoids numbers too small to store.

Real result for the question above: **decision, 100%**. Its strongest clues versus *factual* were `i`, `<num>`, `a`, `am` and `am_i`.

---

### ④ Do the documents cover this question? (evidence check)

**In plain words.** Before calling the AI, the code asks: *what share of the question's important words appear in the top 3 pieces?* Rare words count more. Below **0.30** it stops with "not in the knowledge base", so the AI never gets a chance to invent an answer.

**In the code.** `assess()` ([evidence.py:37-65](app/system1/evidence.py#L37-L65)):

```python
terms = {t for t in result.query_terms
         if not t[0].isdigit() and t not in QUESTION_PHRASING}       # drop numbers + "eligible", "year"...
weights = {t: result.index.idf(t) for t in terms}                    # rare word → heavy weight
total = sum(weights.values())

def coverage(term_sets):                                             # weighted share of words found
    return sum(w * max(covered(t, s) for s in term_sets) for t, w in weights.items()) / total

best = max(coverage([s]) for s in result.passage_terms[:3])          # best single piece
combined = coverage(result.passage_terms[:3])                        # top 3 together
score = 0.5 * best + 0.5 * combined
verdict = "none" if score < 0.30 else "weak" if score < 0.50 else "sufficient"
```

- `covered(t, s)` returns 1.0 for an exact match, 0.8 or 0.7 for a spelling variant, and 0 if the word isn't there.
- `QUESTION_PHRASING` removes words about the *shape* of a question (eligible, difference, year, second…). They say nothing about the topic.

Real calculation for **"What is on the cafeteria menu this Friday?"** (Northbridge):

| Word | Weight (idf) | Found in the top pieces? |
|---|---|---|
| cafeteria | 3.258 (never in the documents) | ✗ |
| menu | 3.258 (never in the documents) | ✗ |
| friday | 2.159 | ✓ (office hours: "Monday to Friday") |

Coverage = 2.159 / (3.258 + 3.258 + 2.159) = **0.249**, which is below 0.30, so the verdict is **none**. The assistant replies "not in the knowledge base" with **0 AI calls**.

For comparison, "What GPA do I need for the Merit Scholarship?" scores **1.0** (every word found).

---

### ⑤ Does it need a yes/no decision? (routing)

**In plain words.** Decision and comparison questions go on to the rule checking in ⑥. Simple factual, policy and procedural questions skip straight to ⑨, which saves an AI call.

**In the code.** [pipeline.py:158](app/pipeline.py#L158):

```python
needs_decisions = classification.label in {"decision", "comparison"} or classification.confidence < 0.55
```

- If the classifier is unsure (less than 55% confident), the question also takes the careful path.

---

### ⑥ Turn the question and rules into a form (AI: decompose)

**In plain words.** The AI gets the question and the top pieces of text, and must fill in a form:

- **facts** the person stated
- **rules** to check, each citing the piece it came from
- **how to combine** the yes/no rules (`all` or `any`)

It must not decide anything, invent rules or guess facts.

**In the code.**

**The form** is defined in [schemas.py](app/schemas.py) as Pydantic classes:

```python
class Fact(BaseModel):        name: str; value: str; kind: "number" | "boolean" | "text"
class Condition(BaseModel):   fact: str; operator: ">=" | "<" | "in" | "is_true" | ...; value: str; source: str
class DecisionSpec(BaseModel):
    id: str; type: "boolean" | "choice" | "score"; question: str
    match: "all" | "any"; conditions: list[Condition]          # boolean
    options: list[Option]                                      # choice: [{label, conditions}] in order
    criteria: list[Criterion]                                  # score:  [{condition, weight}]
class Decomposition(BaseModel):
    facts: list[Fact]; decisions: list[DecisionSpec]; combine: "all" | "any" | "none"
```

**Building the prompt:** `decompose()` ([llm.py:104-110](app/llm.py#L104-L110)):

```python
content = (f"<document_excerpts>\n{_format_passages(passages)}\n</document_excerpts>\n\n"
           f"<question_type>{question_type}</question_type>\n"
           f"<visitor_question>\n{question}\n</visitor_question>")
return self._structured(system=DECOMPOSE_SYSTEM, content=content, schema=Decomposition, effort="medium")
```

- `_format_passages` writes each piece as `[S1] Scholarship_Rules.pdf - page 2` followed by its text.
- `DECOMPOSE_SYSTEM` ([llm.py](app/llm.py)) holds the rules: one yes/no decision per requirement, cite a source, don't invent thresholds, don't guess facts.

**Getting a valid form back:** `_structured()` ([llm.py:287-305](app/llm.py#L287)):

```python
messages = [{"role": "system", "content": system + "Respond with ONLY JSON matching: <schema>"},
            {"role": "user", "content": content}]
for _ in range(2):                                   # at most 2 tries
    text = self._chat(messages)                      # HTTP call to Gemini
    try:
        return schema.model_validate(extract_json(text))    # parse + check the shape
    except (ValueError, ValidationError) as exc:
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That response was invalid: {exc}. Reply again..."}]
raise LLMError("did not return valid structured output")
```

**What comes back** for our question (real output from the Gemini run):

```json
{
  "facts": [
    {"name": "is_full_time", "value": "true"}, {"name": "year_of_study", "value": "2"},
    {"name": "cumulative_gpa", "value": "3.72"}, {"name": "has_academic_misconduct", "value": "false"},
    {"name": "athletic_scholarship_amount", "value": "4000"}
  ],
  "decisions": [
    {"id": "D1", "type": "boolean", "conditions": [{"fact": "is_full_time", "operator": "is_true", "source": "S1"}]},
    {"id": "D2", "type": "boolean", "conditions": [{"fact": "year_of_study", "operator": "in", "value": "2,3,4"}]},
    {"id": "D3", "type": "boolean", "conditions": [{"fact": "cumulative_gpa", "operator": ">=", "value": "3.5"}]},
    {"id": "D4", "type": "boolean", "conditions": [{"fact": "has_academic_misconduct", "operator": "is_false"}]},
    {"id": "D5", "type": "boolean", "conditions": [{"fact": "athletic_scholarship_amount", "operator": "<", "value": "5000"}]},
    {"id": "D6", "type": "choice", "options": [
        {"label": "Gold tier",   "conditions": [{"fact": "cumulative_gpa", "operator": ">=", "value": "3.85"}]},
        {"label": "Silver tier", "conditions": [{"fact": "cumulative_gpa", "operator": ">=", "value": "3.70"}]},
        {"label": "Bronze tier", "conditions": [{"fact": "cumulative_gpa", "operator": ">=", "value": "3.50"}]}]}
  ],
  "combine": "all"
}
```

**The HTTP call itself:** `_chat()` / `_post()` ([llm.py:234-278](app/llm.py#L234-L278)):

```python
for model in [self.model, *self.fallback_models]:     # gemini-flash-latest, then gemini-flash-lite-latest
    for attempt in range(3):                           # retry after waiting 1.5 s, then 4 s
        r = POST {base_url}/chat/completions {"model": model, "messages": ..., "temperature": 0,
                                              "response_format": {"type": "json_object"}}
        if r.status_code not in {429, 500, 502, 503, 504}: break    # not a temporary error
    else:
        continue                                       # still overloaded → try the backup model
    return r.json()["choices"][0]["message"]["content"]
```

---

### ⑦ Check every rule (decision models)

**In plain words.** Plain code checks each rule, all at the same time. Every check answers **yes**, **no** or **unknown** (unknown = the person didn't give that fact). There are three kinds of decision:

- **Yes/No (Boolean):** does the rule pass?
- **Pick one (Choice):** which category applies? Options are checked in order and the first that fits wins.
- **Score:** how strongly is the outcome supported? A weighted share of criteria met.

**In the code.**

**Running them in parallel:** `run_decisions()` ([decisions.py:224-227](app/system1/decisions.py#L224-L227)):

```python
table = build_fact_table(facts)                          # {"cumulative_gpa": Fact(...), ...}
return await asyncio.gather(*(asyncio.to_thread(_run_one, s, table) for s in specs))
```

`_run_one()` ([decisions.py:213-221](app/system1/decisions.py#L213-L221)) picks the evaluator for `spec.type` and wraps it in `try/except`. If one rule is malformed, that one becomes an `error` result and the others still run.

**Checking one condition:** `evaluate_condition()` ([decisions.py:64-101](app/system1/decisions.py#L64-L101)):

```python
fact = facts.get(normalize_name(cond.fact))
if fact is None:
    return UNKNOWN                                         # person didn't tell us → unknown
if op in ("is_true", "is_false"):
    b = to_bool(fact.value)                                # "true"/"yes"/"1" → True
    return TRUE if b == (op == "is_true") else FALSE
if op in ("in", "not_in"):
    allowed = {v.strip().lower() for v in cond.value.split(",")}   # "2,3,4" → {"2","3","4"}
    hit = fact.value.strip().lower() in allowed
    return TRUE if hit == (op == "in") else FALSE
left, right = to_number(fact.value), to_number(cond.value)         # "$4,000" → 4000.0
ok = {">=": left >= right, ">": left > right, "<=": left <= right, "<": left < right, ...}[op]
return TRUE if ok else FALSE
```

**Yes/No:** `evaluate_boolean()` ([decisions.py:143-153](app/system1/decisions.py#L143-L153)):

```python
results = [evaluate_condition(c, facts) for c in spec.conditions]
outcome = (tri_all if spec.match == "all" else tri_any)([r.result for r in results])
```

**Pick one:** `evaluate_choice()` ([decisions.py:156-185](app/system1/decisions.py#L156-L185)):

```python
for option in spec.options:                                  # Gold, then Silver, then Bronze
    verdict = tri_all([evaluate_condition(c, facts) for c in option.conditions])
    if verdict == TRUE and blocked_by is None:
        selected = option.label; break                       # first option that fits wins
    if verdict == UNKNOWN and blocked_by is None:
        blocked_by = option.label                            # can't rule it out → answer "unknown"
```

**Score:** `evaluate_score()` ([decisions.py:188-203](app/system1/decisions.py#L188-L203)):

```python
got     = sum(w for w, r in zip(weights, results) if r.result == TRUE)    / total   # share met
unknown = sum(w for w, r in zip(weights, results) if r.result == UNKNOWN) / total   # still open
band = "strong" if got >= 0.75 else "moderate" if got >= 0.4 else "weak"
```

For example, with criteria weighted 0.5, 0.3 and 0.2, where the first two pass and the third is unknown: `got = 0.8` (strong), and "could reach 1.00 with missing facts".

**Our question, evaluated:**

| Rule | Fact value | Check | Result |
|---|---|---|---|
| D1 | is_full_time = true | is_true | ✅ TRUE |
| D2 | year_of_study = 2 | in {2, 3, 4} | ✅ TRUE |
| D3 | cumulative_gpa = 3.72 | ≥ 3.5 | ✅ TRUE |
| D4 | has_academic_misconduct = false | is_false | ✅ TRUE |
| D5 | athletic_scholarship_amount = 4000 | < 5000 | ✅ TRUE |
| D6 | cumulative_gpa = 3.72 | Gold ≥ 3.85 ✗ → Silver ≥ 3.70 ✓ | **Silver tier** |

The six checks together took about **2 milliseconds**. For comparison, the two AI calls took about **15 seconds**.

---

### ⑧ Combine the results

**In plain words.** "All must pass": any NO means **No**; all YES means **Yes**; otherwise **Need more information**, and the reply lists the missing facts.

**In the code.** The three-valued AND and OR ([decisions.py:104-115](app/system1/decisions.py#L104-L115)):

```python
def tri_all(values):                 # "all must pass"
    if FALSE in values:   return FALSE           # one failure is enough
    if UNKNOWN in values: return UNKNOWN         # can't be sure yet
    return TRUE

def tri_any(values):                 # "any one is enough"
    if TRUE in values:    return TRUE
    if UNKNOWN in values: return UNKNOWN
    return FALSE
```

`combine()` ([combiner.py:23-54](app/combiner.py#L23-L54)):

```python
booleans = [Tri(r.outcome) for r in ok if r.type == "boolean"]
if failed and decomposition.combine != "none" and any(r.type == "boolean" for r in failed):
    overall = "error"                                   # a required check broke → don't guess
elif decomposition.combine == "none" or not booleans:
    overall = "informational"                           # (or "insufficient_information" if a choice is unknown)
else:
    verdict = (tri_all if decomposition.combine == "all" else tri_any)(booleans)
    overall = {TRUE: "yes", FALSE: "no", UNKNOWN: "insufficient_information"}[verdict]
missing = sorted({m for r in ok for m in r.missing_facts}) if overall != "no" else []
```

Our question: `tri_all([TRUE, TRUE, TRUE, TRUE, TRUE])` → **TRUE** → `overall = "yes"`.

Two variations:
- **$6,000 sports scholarship:** D5 is FALSE → `tri_all` → FALSE → **"no"**.
- **GPA left out:** D3 is UNKNOWN and D6 is unknown → **"insufficient_information"**, with `missing_facts = ["cumulative_gpa"]`, so the reply asks for the GPA.

---

### ⑨ Write the answer (AI: generate)

**In plain words.** The AI writes the reply from the pieces of text plus the results, which it is told are **final**. It cites pieces as `[S1]` and reports `grounded=false` if the text doesn't contain the answer. The code then turns the citations into document names and pages.

**In the code.**

The results are first turned into a readable summary for the AI by `_decision_context()` ([pipeline.py:67](app/pipeline.py#L67)):

```python
{"overall_outcome": "yes", "combination_rule": "all", "missing_facts": [],
 "decisions": [{"question": "...", "outcome": "true",
                "checks": [{"rule": "GPA of at least 3.5", "visitor_value": "3.72",
                            "required": ">= 3.5", "result": "true", "source": "S1"}]}, ...]}
```

Then `generate()` ([llm.py:112-121](app/llm.py#L112-L121)) sends the pieces of text, that summary and the question, with `ANSWER_SYSTEM`: *"results are authoritative… cite [S#]… set grounded=false if not covered"*. It returns `GeneratedAnswer{answer, sources_used, grounded}`.

**Turning citations into sources** ([pipeline.py:205-215](app/pipeline.py#L205-L215)):

```python
if not generated.grounded:
    return finish("unable_to_answer", NOT_FOUND_MESSAGE, [])          # second safety net
valid_ids = {p.sid for p in passages}                                  # S1..S6 we actually sent
cited = (set(generated.sources_used) | set(re.findall(r"\[(S\d+)\]", generated.answer))) & valid_ids
cited |= decision_sources & valid_ids                                  # pages the rules came from
return finish(status, generated.answer, _references(passages, cited))
```

- `& valid_ids` throws away any source the AI invented. Only pieces we really gave it can be cited.
- `_references()` ([pipeline.py:86](app/pipeline.py#L86)) merges them into one entry per (document, page): `Scholarship_Rules.pdf — Page 2`.
- `finish()` calls `_log()` ([pipeline.py:100](app/pipeline.py#L100)), which saves the question and its status for the dashboard's usage counts.

---

### The whole orchestrator in one place

This is `answer_question()` ([pipeline.py:109-215](app/pipeline.py#L109-L215)) with the error handling and trace lines removed. It is the core algorithm in about 25 lines:

```python
async def answer_question(org_id, org_name, question):
    if reply := small_talk_reply(question, org_name):                          # "hi" → canned reply
        return finish("answered", reply, [])

    retrieval, classification = await asyncio.gather(                          # ② + ③ in parallel
        asyncio.to_thread(retrieve, org_id, question, 8),
        asyncio.to_thread(classifier.predict, question))

    if assess(retrieval).verdict == "none":                                    # ④
        return finish("unable_to_answer", NOT_FOUND_MESSAGE, [])

    decision_context = None
    if classification.label in {"decision", "comparison"} or classification.confidence < 0.55:   # ⑤
        decomposition = llm.decompose(question, passages, classification.label)                   # ⑥
        results = await run_decisions(decomposition.decisions, decomposition.facts)               # ⑦
        combined = combine(decomposition, results)                                                # ⑧
        decision_context = _decision_context(decomposition, results, combined)

    generated = llm.generate(question, passages, decision_context)                               # ⑨
    if not generated.grounded:
        return finish("unable_to_answer", NOT_FOUND_MESSAGE, [])
    return finish("answered", generated.answer, _references(passages, cited))
```

Which questions take which path:

| Question | Path | AI calls |
|---|---|---|
| "Hi" | small talk | 0 |
| "What's on the cafeteria menu?" | ② ③ ④ → stop | 0 |
| "What is Manav's gpa?" | ② ③ ④ ⑤ → ⑨ | 1 |
| "Am I eligible…?" | ② through ⑨ | 2 |

In the chat UI, **"How this answer was produced"** shows this `trace`: each step, whether it was AI or code, its time, and every rule result.

---

## Part 2b: Conversation memory (continuing a chat)

### In plain words

The assistant remembers each chat, so visitors don't have to repeat themselves:

```
Visitor: Am I eligible for the Merit Scholarship? I'm a third-year full-time student.
Bot:     I still need your GPA.                       ← GPA check = unknown; remembers "waiting for GPA"
Visitor: 3.9
Bot:     Yes, you're eligible!                        ← year + full-time come from memory
Visitor: How do I apply for it?
Bot:     Log in to the Student Portal…                ← "it" = the Merit Scholarship
```

Three things are remembered per conversation:
1. **The last few messages**, so the AI understands follow-ups like "it" or "what about Gold tier?"
2. **The facts the visitor gave** (GPA, year…), reused by the rule checkers later.
3. **What we're waiting for**: if the last answer asked for missing facts, the next message is treated as the answer.

### In the code

**Storage:** two tables (`conversations`, `messages`) and [app/conversations.py](app/conversations.py):

```python
conv = load_or_start(org_id, conversation_id)   # resume only if the id exists, is < 7 days old AND
                                                # belongs to this org; otherwise start a new one
history = format_history(conv.history)          # last 6 messages, as "Visitor: …" / "Assistant: …"
...
save_turn(conv, question, answer, status, references)   # store both messages + facts + pending
```

**Follow-up search** ([pipeline.py](app/pipeline.py), `answer_question()`):

```python
retrieval = retrieve(org_id, question)          # first, search the message on its own
evidence = assess(retrieval)
if conv.history and evidence.verdict != "sufficient" and (conv.pending or len(evidence.topical_terms) <= 2):
    context_query = " ".join([*conv.recent_user_questions(2), question])   # e.g. "Am I eligible…? 3.9"
    ctx = retrieve(org_id, context_query)
    if assess(ctx).score > evidence.score:
        retrieval = ctx                          # the conversation made the search better → use it
```

A short follow-up like "3.9" matches nothing on its own, so it's searched together with the previous question. A complete new question ("How do I appeal?") is searched alone, so old context doesn't pollute it.

**Waiting decisions and fact memory:**

```python
needs_decisions = label in {"decision", "comparison"} or confidence < 0.55 or conv.pending is not None
decomposition = llm.decompose(question, passages, label, history, list(conv.facts.values()))
conv.facts = remember_facts(conv.facts, decomposition.facts)          # old facts + new (new wins)
decomposition.facts = [Fact(**f) for f in conv.facts.values()]       # evaluators see everything
...
if combined.overall == "insufficient_information":
    conv.pending = {"question": ..., "missing_facts": combined.missing_facts}   # ask, and remember why
```

**The chat UI** ([static/assistant.js](static/assistant.js)) stores only the conversation id, in `localStorage` per assistant. On reload it calls `GET /api/assistant/{token}/conversations/{id}` to redraw the chat. **New chat** forgets the id.

**Isolation:** every query uses `WHERE id = ? AND org_id = ?`, so another organization's conversation id is simply treated as unknown. This is tested in [tests/test_memory.py](tests/test_memory.py).

---

## Part 3: When something goes wrong

| Problem | What happens | Code |
|---|---|---|
| AI overloaded (503) or rate limited (429) | Retry after 1.5 s and 4 s, then switch to the backup model | `_chat()` [llm.py:249](app/llm.py#L249) |
| AI returns a badly-shaped form | Told what was wrong, asked once more | `_structured()` [llm.py:287](app/llm.py#L287) |
| AI completely unavailable | Top 3 passages shown with their pages; status `error` | [pipeline.py:197-203](app/pipeline.py#L197-L203) |
| Turning the question into a form fails | Answers from the text anyway; status `needs_review` | [pipeline.py:171-174](app/pipeline.py#L171-L174) |
| One rule check crashes | Only that rule becomes `error`; no verdict; `needs_review` | `_run_one()` [decisions.py:213](app/system1/decisions.py#L213) |
| Bad upload | A specific message (too big, too many pages, not a PDF…) | `ingest_pdf()` [documents.py](app/documents.py) |

---

## Part 4: Keeping organizations separate

| Rule | Where in the code |
|---|---|
| Dashboard calls need a login cookie, and use that cookie's organization only | `require_admin()` [auth.py:100](app/auth.py#L100) |
| The assistant token can only ask questions; admin calls ignore it | `require_admin()` reads only the session cookie |
| Search loads only the asking organization's text | `WHERE c.org_id = ?` in `get_index()` [retrieval.py:151](app/retrieval.py#L151) |
| Document delete/replace needs `id AND org_id` to match | `delete_document()` / `ingest_pdf()` [documents.py](app/documents.py) |
| PDFs are stored outside the public folder and never served | `data/uploads/<org_id>/` |
| All of the above is tested | [tests/test_isolation.py](tests/test_isolation.py) |

---

## Part 5: Which file does what

| File | Role | Key functions |
|---|---|---|
| [app/main.py](app/main.py) | Web routes (the front door) | `ask()`, `upload_document()`, `register()`, `login()` |
| [app/auth.py](app/auth.py) | Logins, sessions, tokens | `org_for_assistant_token()`, `require_admin()`, `create_session()` |
| [app/pipeline.py](app/pipeline.py) | **The conductor** (steps ①–⑨) | `answer_question()`, `_decision_context()`, `_references()`, `remember_facts()` |
| [app/conversations.py](app/conversations.py) | Conversation memory | `load_or_start()`, `save_turn()`, `transcript()`, `format_history()` |
| [app/retrieval.py](app/retrieval.py) | BM25 search | `tokenize()`, `get_index()`, `BM25Index.score()`, `expand()`, `retrieve()` |
| [app/system1/classifier.py](app/system1/classifier.py) | Question type | `features()`, `QuestionClassifier.predict()` |
| [app/system1/evidence.py](app/system1/evidence.py) | Evidence check | `assess()` |
| [app/llm.py](app/llm.py) | Talking to the AI | `decompose()`, `generate()`, `ocr()`, `_structured()`, `_chat()` |
| [app/schemas.py](app/schemas.py) | The form's shape | `Fact`, `Condition`, `DecisionSpec`, `Decomposition`, `GeneratedAnswer` |
| [app/system1/decisions.py](app/system1/decisions.py) | Rule checks | `evaluate_condition()`, `evaluate_boolean/choice/score()`, `run_decisions()`, `tri_all/any()` |
| [app/combiner.py](app/combiner.py) | Combining | `combine()` |
| [app/documents.py](app/documents.py) | Uploads | `ingest_pdf()`, `extract_pages()`, `ocr_pages()`, `chunk_page()` |
| [app/db.py](app/db.py), [app/schema.sql](app/schema.sql) | Database | `get_conn()`, `init_db()` |
| [static/assistant.js](static/assistant.js) | Chat window | `ask()`, `renderMarkdown()`, `renderTrace()` |
| [static/dashboard.js](static/dashboard.js) | Dashboard | `uploadFiles()`, `loadDocs()`, `loadUsage()` |
| [static/embed.js](static/embed.js) | Chat bubble for other websites | (runs once and adds the iframe) |

---

## Part 6: The web API

**Organizers** (need to be logged in): `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`, `GET/POST /api/admin/documents`, `POST /api/admin/documents/{id}/replace`, `DELETE /api/admin/documents/{id}`, `GET /api/admin/assistant`, `POST /api/admin/assistant/regenerate`, `GET /api/admin/usage`.

**Visitors** (token only): `GET /a/{token}` (chat page), `GET /api/assistant/{token}` (organization name only), `GET /api/assistant/{token}/conversations/{id}` (restore a chat), `POST /api/ask` (send `conversation_id` to continue a chat).

```http
POST /api/ask
{"token": "ast_qYiiepw4T1k9...", "question": "What is Manav's gpa?"}
```

```json
{
  "status": "answered",
  "answer": "Manav's CGPA is 8.89/10 [S1].",
  "references": [{"document": "Manav_Narang_Resume (3).pdf", "page": 1, "snippet": "...CGPA: 8.89/10..."}],
  "error": null,
  "latency_ms": 4210,
  "trace": {"classification": {...}, "evidence": {...}, "route": {...}, "stages": [...]}
}
```

`status` is one of: `answered` · `unable_to_answer` (not in the knowledge base) · `needs_review` (answered, but part of the process failed) · `error` (AI unavailable, so passages are shown instead).

---

## Glossary

| Term | Meaning |
|---|---|
| **LLM / AI** | The language model (Gemini). Only used to understand and write text, and to read scanned pages. |
| **System 1 model** | A fast, predictable, non-AI component: the classifier, the evidence check and the rule checkers. |
| **Chunk** | A paragraph-sized piece of one page, stored with its organization and page number. |
| **BM25** | Keyword-search scoring: rare words count more, repeats have diminishing returns, long pieces are evened out. |
| **IDF** | "Inverse document frequency": the rarity weight of a word. Fewer pieces containing it means a higher weight. |
| **Naive Bayes** | A classifier that learns from labelled examples how often each word appears in each category. |
| **Three-valued logic** | TRUE / FALSE / UNKNOWN. UNKNOWN = a fact wasn't given, which leads to a follow-up question, not a guess. |
| **Tenant** | One organization. Its data is always filtered by `org_id`. |
| **Assistant token** | The random `ast_…` code in the public link. It identifies the organization and can only be used to ask questions. |
| **OCR** | Reading text from a picture of a page. |
