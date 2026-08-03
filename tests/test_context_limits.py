"""Context caps + expanded provider-health classification."""
from __future__ import annotations

from src import config
from src.grading_engine import _is_provider_unhealthy
from src.prompts import build_grading_system_text, cap_text
from src.models import RubricComponent


def test_cap_text_truncates_and_marks():
    long = "x" * 10_000
    out = cap_text(long, 6000)
    assert len(out) <= 6000 + 40
    assert "truncated" in out


def test_cap_text_leaves_short_alone():
    assert cap_text("short", 6000) == "short"


def test_system_text_bounds_each_sample():
    huge = "WORD " * 5000  # ~25k chars
    text = build_grading_system_text(
        task_text=huge,
        rubric_text=huge,
        components=[RubricComponent(name="C", weight=100, description="")],
        reference_samples={"HD": huge, "DI": huge, "AVG": huge, "PASS": huge},
    )
    # Each capped piece must be far below the raw 25k it was handed.
    assert text.count("truncated") >= 5  # task + rubric + 4 samples
    # Sanity: overall context is bounded well under the uncapped ~150k chars.
    # Measure the fixed scaffolding (persona + headings + JSON instruction)
    # with empty inputs, so the bound survives prompt-wording changes.
    overhead = len(
        build_grading_system_text(
            task_text="",
            rubric_text="",
            components=[RubricComponent(name="C", weight=100, description="")],
            reference_samples={},
        )
    )
    assert len(text) < (
        config.MAX_TASK_CHARS + config.MAX_RUBRIC_CHARS
        + 4 * config.MAX_SAMPLE_CHARS + overhead + 1000
    )


def test_health_classification_request_too_large():
    class Boom(Exception):
        status_code = 413

    assert _is_provider_unhealthy(Boom())


def test_health_classification_invalid_api_key_message():
    # Gemini returns HTTP 400 for a bad key — caught by message, not status.
    class BadReq(Exception):
        status_code = 400

    exc = BadReq("Error: API Key not found. Please pass a valid API key.")
    assert _is_provider_unhealthy(exc)


def test_health_classification_tpm_limit_message():
    exc = Exception("Request too large ... tokens per minute (TPM): Limit 12000")
    assert _is_provider_unhealthy(exc)


def test_health_classification_ignores_plain_content_error():
    # A JSON/validation problem for one report must NOT trip provider fallback.
    class ValidationError(Exception):
        pass

    assert not _is_provider_unhealthy(ValidationError("Expecting value: line 1"))
