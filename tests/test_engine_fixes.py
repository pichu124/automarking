"""Regression tests for LLM-pipeline defects: health misclassification,
context-overflow trimming, empty summaries, JSON-mode retries, per-model
request parameters, <think> blocks."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import ValidationError

from src import config
from src.document_parser import ParsedDocument
from src.grading_engine import (
    GradingSession,
    ProviderSpec,
    _context_overflow_keep_chars,
    _is_provider_unhealthy,
    grade_batch,
    grade_batch_with_fallback,
    summarize_samples_with_fallback,
)
from src.llm_client import GradingLLM
from src.models import ComponentGrade, GradingResult, RubricComponent
from src.prompts import build_grading_user_message
from src.providers import PROVIDER_PRESETS, ModelOutputError, _extract_json, build_provider

COMPONENTS = [RubricComponent(name="Authentication controls", weight=100.0, description="")]
GRADE_JSON = (
    '{"student_identifier": "s", "components": [{"name": "Authentication controls", '
    '"score": 70, "max_score": 100, "notes": ["n"]}], "total": 70}'
)


def _ok(name: str) -> GradingResult:
    return GradingResult(
        student_identifier=name,
        components=[ComponentGrade(name="C", score=70.0, max_score=100.0, notes=["n"])],
        total=70.0,
    )


def _session() -> GradingSession:
    return GradingSession(
        task_text="task", rubric_text="rubric", components=COMPONENTS, reference_samples={}
    )


def _validation_error() -> ValidationError:
    # Model omitted "notes" on a component named after an auth topic: pydantic
    # quotes the model output, so the message contains "authentication".
    try:
        GradingResult.model_validate({
            "student_identifier": "s",
            "components": [{"name": "Authentication controls", "score": 5, "max_score": 10}],
            "total": 5,
        })
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class FakeChat:
    """Stands in for ``openai.OpenAI().chat.completions``; records every call."""

    def __init__(self, content="", finish_reason="stop", errors=()):
        self.content = content
        self.finish_reason = finish_reason
        self.errors = list(errors)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)]
        )


def _openai_error(cls, status: int):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.test/v1"))
    return cls("boom", response=response, body=None)


# --- 1. Health classification: content errors stay per-report; 404 falls back -

def test_validation_error_naming_authentication_is_not_a_provider_failure():
    exc = _validation_error()
    assert "authentication" in str(exc).lower()
    assert not _is_provider_unhealthy(exc)

    class Grader:
        label = "primary"

        def grade_report(self, *, source_name, **_):
            if source_name == "bad.docx":
                raise exc
            return _ok(source_name)

    docs = [ParsedDocument("bad.docx", "x"), ParsedDocument("good.docx", "y")]
    res = grade_batch_with_fallback(
        [ProviderSpec("p", label="primary"), ProviderSpec("q", label="secondary")],
        _session(), docs, builder=lambda spec: Grader(),
    )
    assert res.provider_label == "primary"
    assert [r.source_name for r in res.reports] == ["good.docx"]
    assert len(res.errors) == 1 and res.errors[0].startswith("bad.docx")


def test_claude_unparseable_output_is_a_model_output_error():
    llm = GradingLLM(api_key="test-key")
    llm.client = SimpleNamespace(
        messages=SimpleNamespace(parse=lambda **_: SimpleNamespace(parsed_output=None))
    )
    with pytest.raises(ModelOutputError) as grading:
        llm.grade_report(
            task_text="t", rubric_text="r", components=COMPONENTS, reference_samples={},
            source_name="quota_analysis.docx", report_text="body",
        )
    assert not _is_provider_unhealthy(grading.value)
    with pytest.raises(ModelOutputError):
        llm.structure_rubric("rubric about quota billing")


def test_model_not_found_404_is_a_provider_failure():
    assert _is_provider_unhealthy(_openai_error(openai.NotFoundError, 404))


# --- 2. Context overflow trims by the excess, from the length actually sent ---

def test_overflow_trims_by_excess_not_by_completion_ratio():
    openai_style = Exception(
        "This model's maximum context length is 128000 tokens. However, you "
        "requested 140000 tokens (124000 in the messages, 16000 in the completion)."
    )
    keep = _context_overflow_keep_chars(openai_style, 480_000)
    # 12,000 tokens over => drop ~48k-55k chars, keep the rest (not ~10%).
    assert 420_000 <= keep <= 480_000 - 12_000 * 4

    anthropic_style = Exception("prompt is too long: 1600280 tokens > 1000000 maximum")
    keep = _context_overflow_keep_chars(anthropic_style, 6_400_000)
    assert 3_500_000 < keep <= 6_400_000 - 600_280 * 4


def test_overflow_retry_shrinks_the_capped_text_actually_sent(monkeypatch):
    monkeypatch.setattr(config, "MAX_REPORT_CHARS", 40_000)
    limit = 9_000
    sent_lengths = []

    class Grader:
        label = "primary"

        def grade_report(self, *, source_name, report_text, **_):
            sent = build_grading_user_message(source_name, report_text)
            sent_lengths.append(len(sent))
            tokens = len(sent) // 4
            if tokens > limit:
                raise Exception(f"prompt is too long: {tokens} tokens > {limit} maximum")
            return _ok(source_name)

    res = grade_batch(Grader(), _session(), [ParsedDocument("big.docx", "word " * 20_000)])
    assert res.errors == []
    assert len(res.reports) == 1
    assert sent_lengths[-1] < sent_lengths[0]
    assert any("context window" in n for n in res.notes)


# --- 3. An empty summary never silently replaces the sample -------------------

def test_openai_compatible_empty_completion_raises():
    provider = build_provider("openai", api_key="test-key")
    provider.client = FakeChat(content="", finish_reason="length")
    with pytest.raises(ModelOutputError):
        provider.complete_text("system", "user")


def test_claude_completion_with_only_thinking_raises():
    llm = GradingLLM(api_key="test-key")
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="...")], stop_reason="max_tokens"
    )
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    with pytest.raises(ModelOutputError):
        llm.complete_text("system", "user")


def test_empty_summary_falls_back_to_capped_sample_with_note(monkeypatch):
    monkeypatch.setattr(config, "MAX_SAMPLE_CHARS", 5_000)

    class Grader:
        label = "primary"

        def complete_text(self, system, user, max_tokens=1200):
            if "High Distinction" in user:
                raise ModelOutputError("model returned an empty completion")
            return "DI profile"

    samples = {"HD": "H" * 9_000, "DI": "D" * 9_000}
    profiles, used = summarize_samples_with_fallback(
        [ProviderSpec("p", label="primary"), ProviderSpec("q", label="secondary")],
        samples, COMPONENTS, builder=lambda spec: Grader(),
    )
    assert used == "primary"
    assert profiles["DI"] == "DI profile"
    hd = profiles["HD"]
    assert "H" * 4_000 in hd                 # the original sample is kept...
    assert hd.count("H") <= 5_000            # ...capped like any raw sample
    assert hd.startswith("[") and "summary" in hd.lower()  # with a visible note


# --- 4. OpenAI-compatible requests: JSON-mode retry, per-model params, <think> -

def test_json_mode_not_retried_on_auth_error():
    provider = build_provider("openai", api_key="test-key")
    provider.client = FakeChat(errors=[_openai_error(openai.AuthenticationError, 401)])
    with pytest.raises(openai.AuthenticationError):
        provider.structure_rubric("rubric")
    assert len(provider.client.calls) == 1


def test_json_mode_retried_without_response_format_on_bad_request():
    provider = build_provider("openai", api_key="test-key")
    provider.client = FakeChat(
        content='{"components": [{"name": "A", "weight": 100, "description": ""}]}',
        errors=[_openai_error(openai.BadRequestError, 400)],
    )
    rubric = provider.structure_rubric("rubric")
    assert [c.name for c in rubric.components] == ["A"]
    assert "response_format" in provider.client.calls[0]
    assert "response_format" not in provider.client.calls[1]


def _grade(provider) -> dict:
    provider.client = FakeChat(content=GRADE_JSON)
    provider.grade_report(
        task_text="t", rubric_text="r", components=COMPONENTS, reference_samples={},
        source_name="s.docx", report_text="body",
    )
    return provider.client.calls[-1]


def test_deepseek_output_tokens_clamped_to_preset_cap():
    sent = _grade(build_provider("deepseek", api_key="test-key"))
    assert sent["max_tokens"] == 8192
    # deepseek-reasoner allows 64K and spends its chain of thought from the
    # same budget, so the chat model's cap must not truncate it.
    sent = _grade(build_provider("deepseek", model="deepseek-reasoner", api_key="test-key"))
    assert sent["max_tokens"] == config.GRADING_MAX_TOKENS


def test_openai_reasoning_models_use_max_completion_tokens_without_temperature():
    for key, model in (("openai", "o3-mini"), ("openrouter", "openai/gpt-5-mini")):
        sent = _grade(build_provider(key, model=model, api_key="test-key"))
        assert sent["max_completion_tokens"] == config.GRADING_MAX_TOKENS
        assert "max_tokens" not in sent and "temperature" not in sent

    sent = _grade(build_provider("openai", model="gpt-4o-mini", api_key="test-key"))
    assert sent["max_tokens"] == config.GRADING_MAX_TOKENS and "temperature" in sent


def test_extract_json_ignores_think_block_with_braces():
    text = '<think>Shape is {"total": n}; check {braces}.</think>\n{"total": 65}'
    assert _extract_json(text) == {"total": 65}


# --- 5. Retired Gemini model -------------------------------------------------

def test_gemini_default_model_is_not_retired():
    assert PROVIDER_PRESETS["gemini"].default_model == "gemini-3.6-flash"
