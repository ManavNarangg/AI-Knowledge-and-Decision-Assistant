"""OpenAI-compatible backend (Groq / Gemini / Ollama) tested against a mocked HTTP transport."""
import json

import httpx
import pytest

from app import llm as llm_module
from app.llm import LLMError, OpenAICompatibleLLM, UnavailableLLM, extract_json
from app.retrieval import Passage
from app.schemas import Decomposition, GeneratedAnswer

PASSAGES = [Passage("S1", 1, "d", "Rules.pdf", 2, "GPA of at least 3.5 is required.", 1.0)]


def backend(responses: list, seen: list | None = None, fallbacks=None) -> OpenAICompatibleLLM:
    """Serve canned (status, body) responses in order; record request bodies."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        status, body = queue.pop(0)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return OpenAICompatibleLLM("test-model", "https://llm.test/v1", "key", "groq", fallback_models=fallbacks,
                               transport=httpx.MockTransport(handler), backoff=(0, 0))


def reply(content: str) -> tuple[int, dict]:
    return 200, {"choices": [{"message": {"content": content}}]}


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_generate_parses_and_validates():
    seen = []
    llm = backend([reply('{"answer": "Yes [S1]", "sources_used": ["S1"], "grounded": true}')], seen)
    out = llm.generate("Is 3.6 enough?", PASSAGES, None)
    assert isinstance(out, GeneratedAnswer) and out.grounded and out.sources_used == ["S1"]
    body = seen[0]
    assert body["model"] == "test-model" and body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][0]["content"] and "[S1] Rules.pdf - page 2" in body["messages"][1]["content"]


def test_lenient_defaults_accept_minimal_decomposition():
    llm = backend([reply(json.dumps({"facts": [{"name": "gpa", "value": "3.6"}], "decisions": [
        {"id": "D1", "type": "boolean", "question": "GPA ok?",
         "conditions": [{"fact": "gpa", "operator": ">=", "value": "3.5", "source": "S1"}]}], "combine": "all"}))])
    d = llm.decompose("GPA 3.6, eligible?", PASSAGES, "decision")
    assert isinstance(d, Decomposition) and d.decisions[0].match == "all" and d.decisions[0].options == []


def test_invalid_output_gets_one_corrective_retry():
    seen = []
    llm = backend([reply('{"answer": "missing grounded field"}'),
                   reply('{"answer": "ok [S1]", "sources_used": ["S1"], "grounded": true}')], seen)
    assert llm.generate("q", PASSAGES, None).answer == "ok [S1]"
    assert "invalid" in seen[1]["messages"][-1]["content"]


def test_persistent_invalid_output_raises():
    llm = backend([reply("not json"), reply("still not json")])
    with pytest.raises(LLMError, match="valid structured output"):
        llm.generate("q", PASSAGES, None)


def test_falls_back_when_json_mode_unsupported():
    seen = []
    llm = backend([(400, "response_format is not supported by this model"),
                   reply('{"answer": "a", "grounded": false}')], seen)
    assert llm.generate("q", PASSAGES, None).grounded is False
    assert "response_format" in seen[0] and "response_format" not in seen[1]


@pytest.mark.parametrize("responses, message", [
    ([(401, {"error": "x"})], "rejected the API key"),
    ([(404, [{"error": {"message": "retired model"}}])], "retired model"),
    ([(400, {"error": {"message": "bad request"}})], "bad request"),
    ([(429, {})] * 3, "rate limited"),
    ([(503, {})] * 3, "temporarily overloaded"),
])
def test_http_errors_become_user_facing_llm_errors(responses, message):
    with pytest.raises(LLMError, match=message):
        backend(responses).generate("q", PASSAGES, None)


def test_transient_errors_are_retried():
    llm = backend([(503, {}), (429, {}), reply('{"answer": "a", "grounded": true}')])
    assert llm.generate("q", PASSAGES, None).answer == "a"


def test_overloaded_model_falls_back_to_next_model():
    seen = []
    llm = backend([(503, {})] * 3 + [reply('{"answer": "lite", "grounded": true}')], seen, fallbacks=["lite-model"])
    assert llm.generate("q", PASSAGES, None).answer == "lite"
    assert [b["model"] for b in seen] == ["test-model"] * 3 + ["lite-model"]
    assert llm.last_model_used == "lite-model"


def test_provider_resolution(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm_module.settings, "llm_provider", "auto")
    assert llm_module.resolve_provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert llm_module.resolve_provider() == "gemini"
    built = llm_module.build_llm()
    assert built.provider == "gemini" and built.model == "gemini-flash-latest"
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert llm_module.resolve_provider() == "groq"


def test_unconfigured_llm_degrades_instead_of_crashing(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm_module.settings, "llm_provider", "auto")
    llm_module.set_llm(None)
    try:
        client = llm_module.get_llm()
        assert isinstance(client, UnavailableLLM)
        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            client.generate("q", PASSAGES, None)
    finally:
        llm_module.set_llm(None)
