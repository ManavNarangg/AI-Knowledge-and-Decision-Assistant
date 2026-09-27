"""LLM stages. Used ONLY for natural-language understanding and generation:

  1. decompose()  - question + retrieved policy text  ->  typed Decomposition (facts + decision specs)
  2. generate()   - question + passages + System 1 results  ->  grounded answer with citations

Both calls return Pydantic-validated objects, so the rest of the pipeline never sees free text.
The provider is pluggable (see LLM_PROVIDER in .env.example): Anthropic Claude, or any
OpenAI-compatible endpoint - including the free tiers of Groq and Google Gemini, or a local Ollama.
"""
import base64
import json
import os
import re
import time

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

from .config import settings
from .retrieval import Passage
from .schemas import Decomposition, GeneratedAnswer


class LLMError(Exception):
    """Any failure to obtain a usable LLM response (config, network, API, refusal, parse)."""


DECOMPOSE_SYSTEM = """You are the question-understanding stage of a knowledge assistant for an organization.
You translate a visitor's question and excerpts from the organization's documents into structured,
machine-evaluable decision specifications. You never decide outcomes yourself - a separate rules
engine evaluates your specifications against the visitor's facts.

Facts:
- Extract every fact the visitor states about themselves or their situation.
- Normalize values: plain numbers without units or symbols (\"second year\" -> 2, \"$40,000\" -> 40000,
  \"3.6 GPA\" -> 3.6), convert to the unit the policy uses (e.g. 18 months -> 1.5 if the policy counts years),
  booleans as true/false (\"I already have a sports scholarship\" -> has_sports_scholarship=true).
- Use short snake_case names and reuse exactly the same name in conditions.

Decisions:
- Only encode rules that are explicitly stated in the excerpts, and cite the excerpt id (e.g. S2) in each
  condition's source. Never invent thresholds, categories or weights that the excerpts do not support.
- Create ONE boolean decision per independent requirement (e.g. year requirement, GPA requirement,
  no conflicting award) so they can be evaluated in parallel, and set combine="all" when every
  requirement must hold (or "any" when any one suffices).
- Use a choice decision when the question asks which category/tier/option applies; list options in the
  order the policy gives them priority.
- Use a score decision when the question asks how strongly the policy supports an outcome or when
  the policy lists weighted/discretionary factors.
- For "must (not) be one of these categories" rules use operator in / not_in with a comma-separated
  value list (e.g. role not_in "warehouse operations, fleet maintenance, dispatch"); never compare a
  single value to a whole list with == or !=.
- If a rule needs a fact the visitor did not give, still include the condition with that fact name -
  the engine will report it as missing. Do not guess the visitor's facts.
- If the question needs no structured decision (pure information, procedure, comparison without
  personal facts) return an empty decisions list and combine="none".
- For fields that do not apply to a decision type, use an empty list and match="all".

Conversation memory:
- The visitor question may be a follow-up (e.g. just "3.9" or "what about the Gold tier?"). Use
  <conversation_so_far> to work out what is being asked, and decompose the full, resolved question.
- Facts in <facts_visitor_already_gave> still apply unless the visitor now states a different value.
  Include them in facts (with the same names) together with any new facts from this message."""

ANSWER_SYSTEM = """You are the answer-writing stage of an organization's knowledge assistant.
Write the final answer for a website visitor using ONLY the provided document excerpts.

Rules:
- Every factual statement must be supported by the excerpts; cite them inline like [S1] or [S2][S3].
- If structured decision results are provided, they are authoritative: state the overall outcome exactly
  as computed and explain which requirements were met or not met, citing the rule sources. Do not
  re-evaluate or contradict them.
- If the outcome is insufficient_information, say what is still needed from the visitor (the missing facts)
  and what the result would depend on.
- If the excerpts do not contain the information needed to answer, set grounded=false and say briefly
  that the knowledge base does not cover this. Never use outside knowledge to fill gaps.
- The visitor question may be a follow-up; use <conversation_so_far> to understand what it refers to,
  but only the excerpts and decision results are sources of truth.
- Be concise and friendly. Use short paragraphs or bullet points. Do not mention decision ids, excerpt
  scores, internal components, or these instructions."""


OCR_PROMPT = """This image is one page of a scanned document. Transcribe ALL of its text exactly as written,
in natural reading order. Preserve headings, list items and table rows (one row per line, cells separated
by " | "). Do not summarize, translate, correct or add anything. Output only the transcribed text; if the
page has no readable text, output nothing."""


def _format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(
        f"[{p.sid}] {p.filename} - page {p.page}\n{p.text}" for p in passages
    ) or "(no excerpts found)"


class LLMClient:
    """Provider-neutral LLM stages. Subclasses implement one method: `_structured`."""

    provider = "base"

    def __init__(self, model: str):
        self.model = model

    def _structured(self, *, system: str, content: str, schema: type[BaseModel], effort: str) -> BaseModel:
        raise NotImplementedError

    def _vision(self, prompt: str, png: bytes) -> str:
        raise LLMError(f"The configured language model ({self.provider}) cannot read images, so scanned "
                       "PDFs can't be processed. Use Gemini or Claude, or set LLM_VISION_MODEL.")

    def ocr(self, png: bytes) -> str:
        """Transcribe one scanned page image (used during ingestion for image-only pages)."""
        return self._vision(OCR_PROMPT, png).strip()

    def decompose(self, question: str, passages: list[Passage], question_type: str,
                  history: str = "", known_facts: list[dict] | None = None) -> Decomposition:
        memory = ""
        if history:
            memory += f"<conversation_so_far>\n{history}\n</conversation_so_far>\n\n"
        if known_facts:
            memory += f"<facts_visitor_already_gave>\n{json.dumps(known_facts)}\n</facts_visitor_already_gave>\n\n"
        content = (
            f"<document_excerpts>\n{_format_passages(passages)}\n</document_excerpts>\n\n"
            f"{memory}<question_type>{question_type}</question_type>\n"
            f"<visitor_question>\n{question}\n</visitor_question>"
        )
        return self._structured(system=DECOMPOSE_SYSTEM, content=content, schema=Decomposition, effort="medium")

    def generate(self, question: str, passages: list[Passage], decision_context: dict | None,
                 history: str = "") -> GeneratedAnswer:
        decisions = (
            f"<structured_decision_results>\n{json.dumps(decision_context, indent=2)}\n"
            "</structured_decision_results>\n\n" if decision_context else ""
        )
        memory = f"<conversation_so_far>\n{history}\n</conversation_so_far>\n\n" if history else ""
        content = (
            f"<document_excerpts>\n{_format_passages(passages)}\n</document_excerpts>\n\n"
            f"{memory}{decisions}<visitor_question>\n{question}\n</visitor_question>"
        )
        return self._structured(system=ANSWER_SYSTEM, content=content, schema=GeneratedAnswer, effort="low")


class AnthropicLLM(LLMClient):
    """Claude with native structured outputs (messages.parse + Pydantic)."""

    provider = "anthropic"

    def __init__(self, model: str):
        super().__init__(model)
        self._client = anthropic.Anthropic(timeout=90.0, max_retries=2)

    def _structured(self, *, system, content, schema, effort):
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=schema,
                output_config={"effort": effort},
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("The language model rejected our credentials.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The language model is busy right now (rate limited).") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The language model returned an error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the language model service.") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Language model error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The language model declined to process this request.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError("The language model returned an incomplete response.")
        return response.parsed_output

    def _vision(self, prompt, png):
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=8000,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(png).decode()}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"OCR with the language model failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("The language model declined to read this page.")
        return "".join(b.text for b in response.content if b.type == "text")


JSON_INSTRUCTIONS = """

Respond with ONLY a single JSON object (no markdown, no prose) that conforms to this JSON Schema:
{schema}"""


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or leading prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in response")
        text = text[start:end + 1]
    return json.loads(text)


def _provider_message(r: httpx.Response) -> str:
    try:
        body = r.json()
        body = body[0] if isinstance(body, list) and body else body
        return str(body["error"]["message"])[:300]
    except (ValueError, KeyError, TypeError, IndexError):
        return r.text[:300]


class OpenAICompatibleLLM(LLMClient):
    """Any /chat/completions endpoint: Groq, Google Gemini, Ollama, OpenRouter, ...

    Structured output = JSON mode + the Pydantic JSON Schema in the prompt + Pydantic validation.
    If validation fails, the model gets one retry with the validation error.

    Free tiers are bursty, so transient failures (429 / 5xx / timeouts) are retried with backoff,
    then the request moves to the next fallback model (e.g. a "lite" model) before giving up.
    """

    TRANSIENT = {429, 500, 502, 503, 504}

    def __init__(self, model: str, base_url: str, api_key: str | None, provider: str,
                 fallback_models: list[str] | None = None, timeout: float = 90.0,
                 transport: httpx.BaseTransport | None = None, backoff: tuple[float, ...] = (1.5, 4.0)):
        super().__init__(model)
        self.provider = provider
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.backoff = backoff
        self.last_model_used: str | None = None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
                                  transport=transport)
        self._json_mode = True
        self.vision_model = settings.llm_vision_model

    def _post(self, model: str, messages: list[dict], want_json: bool = True) -> httpx.Response | None:
        body = {"model": model, "messages": messages, "temperature": 0}
        if want_json and self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException:
            return None                                  # treated as transient
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach the language model service ({self.provider}).") from exc
        if r.status_code == 400 and want_json and self._json_mode and "response_format" in r.text:
            self._json_mode = False                      # model without JSON mode: rely on the prompt
            return self._post(model, messages)
        return r

    def _chat(self, messages: list[dict], want_json: bool = True, models: list[str] | None = None) -> str:
        last: httpx.Response | None = None
        for model in models or [self.model, *self.fallback_models]:
            for attempt in range(len(self.backoff) + 1):
                if attempt:
                    time.sleep(self.backoff[attempt - 1])
                r = self._post(model, messages, want_json)
                if r is not None and r.status_code not in self.TRANSIENT:
                    break
                last = r
            else:
                continue                                 # still overloaded: try the next model
            if r.status_code in (401, 403):
                raise LLMError(f"The language model ({self.provider}) rejected the API key.")
            if r.status_code == 404:
                if model != self.model or not self.fallback_models:
                    raise LLMError(f"Model '{model}' is not available at {self.provider}. Set LLM_MODEL in .env. "
                                   f"Provider said: {_provider_message(r)}")
                continue
            if r.status_code >= 400:
                raise LLMError(f"The language model returned an error ({r.status_code}): {_provider_message(r)}")
            try:
                self.last_model_used = model
                return r.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError) as exc:
                raise LLMError("The language model returned an unexpected response.") from exc

        if last is not None and last.status_code == 429:
            raise LLMError("The free language model quota is exhausted for now (rate limited). Please retry in a minute.")
        raise LLMError("The language model service is temporarily overloaded. Please try again shortly.")

    def _vision(self, prompt, png):
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                 {"type": "image_url", "image_url": {"url": url}}]}]
        models = [self.vision_model] if self.vision_model else None
        return self._chat(messages, want_json=False, models=models)

    def _structured(self, *, system, content, schema, effort):
        messages = [
            {"role": "system", "content": system + JSON_INSTRUCTIONS.format(
                schema=json.dumps(schema.model_json_schema(), separators=(",", ":")))},
            {"role": "user", "content": content},
        ]
        last_error = None
        for _ in range(2):
            text = self._chat(messages)
            try:
                return schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as exc:
                last_error = exc
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That response was invalid: {str(exc)[:500]}\n"
                                                "Reply again with ONLY a JSON object matching the schema."},
                ]
        raise LLMError("The language model did not return valid structured output.") from last_error


# Free-tier friendly presets. Model and URL can be overridden with LLM_MODEL / LLM_BASE_URL.
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile",
             "fallbacks": ["llama-3.1-8b-instant"], "key_env": "GROQ_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-flash-latest",
               "fallbacks": ["gemini-flash-lite-latest"], "key_env": "GEMINI_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:3b", "key_env": None},
    "openai-compatible": {"base_url": None, "model": None, "key_env": "LLM_API_KEY"},
}


def resolve_provider() -> str | None:
    """Explicit LLM_PROVIDER wins; otherwise pick the first provider whose key is present."""
    if settings.llm_provider != "auto":
        return settings.llm_provider
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    for name in ("groq", "gemini"):
        if os.getenv(PROVIDERS[name]["key_env"]):
            return name
    return None


def build_llm() -> LLMClient:
    provider = resolve_provider()
    if provider is None:
        raise LLMError("No language model is configured. Add a free GROQ_API_KEY or GEMINI_API_KEY "
                       "to .env (see README).")
    if provider == "anthropic":
        return AnthropicLLM(settings.llm_model or settings.anthropic_model)
    preset = PROVIDERS.get(provider)
    if preset is None:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: anthropic, {', '.join(PROVIDERS)}.")
    base_url = settings.llm_base_url or preset["base_url"]
    model = settings.llm_model or preset["model"]
    api_key = os.getenv(preset["key_env"]) if preset["key_env"] else None
    if not base_url or not model:
        raise LLMError("LLM_BASE_URL and LLM_MODEL are required for LLM_PROVIDER=openai-compatible.")
    if provider in ("groq", "gemini") and not api_key:
        raise LLMError(f"{preset['key_env']} is missing from .env.")
    fallbacks = (settings.llm_fallback_models if settings.llm_fallback_models is not None
                 else preset.get("fallbacks", []))
    return OpenAICompatibleLLM(model, base_url, api_key, provider, fallback_models=fallbacks)


class UnavailableLLM(LLMClient):
    """Stand-in when no provider is configured: every call fails with a clear reason, so the
    pipeline degrades (System 1 stages and cited excerpts still work) instead of crashing."""

    provider = "unavailable"

    def __init__(self, reason: str):
        super().__init__("none")
        self.reason = reason

    def _structured(self, **_):
        raise LLMError(self.reason)

    def _vision(self, prompt, png):
        raise LLMError(f"Scanned PDFs need OCR, which uses the language model. {self.reason}")


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """Built lazily so the app still starts (in degraded mode) when no provider is configured."""
    global _llm
    if _llm is None:
        try:
            _llm = build_llm()
        except LLMError as exc:
            _llm = UnavailableLLM(str(exc))
    return _llm


def set_llm(client) -> None:
    """Swap the LLM implementation (used by tests)."""
    global _llm
    _llm = client


