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
