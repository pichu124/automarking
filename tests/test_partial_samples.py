"""Grading must work with any subset of the four reference samples,
and each sample slot must accept several files (Word + Excel, say).
"""
from __future__ import annotations

from src.models import RubricComponent
from src.prompts import (
    GRADING_SYSTEM_PROMPT,
    _build_session_context,
    build_grading_system_text,
)

_COMPONENTS = [RubricComponent(name="Analysis", weight=100.0, description="")]


def _context(samples: dict) -> str:
    return _build_session_context("the task", "the rubric", _COMPONENTS, samples)


# --- Prompt handles 0..4 anchors ---------------------------------------------

def test_persona_no_longer_promises_four_samples():
    assert "You will be given four" not in GRADING_SYSTEM_PROMPT
    assert "up to four" in GRADING_SYSTEM_PROMPT
    # Explicit fallback instruction for the no-anchor case.
    assert "If NO samples are supplied" in GRADING_SYSTEM_PROMPT


def test_context_with_all_four_samples_has_no_missing_note():
    ctx = _context({"HD": "hd body", "DI": "di body", "AVG": "avg body", "PASS": "pass body"})
    for label in ("HIGH DISTINCTION", "DISTINCTION", "AVERAGE", "PASS"):
        assert f"=== {label} reference sample ===" in ctx
    assert "No sample was provided" not in ctx
    assert "none provided" not in ctx


def test_context_with_partial_samples_names_the_missing_levels():
    ctx = _context({"HD": "hd body", "PASS": "pass body"})
    assert "=== HIGH DISTINCTION reference sample ===" in ctx
    assert "=== PASS reference sample ===" in ctx
    assert "No sample was provided for: DISTINCTION, AVERAGE." in ctx
    assert "Interpolate" in ctx


def test_context_with_no_samples_directs_to_rubric():
    ctx = _context({})
    assert "none provided" in ctx
    assert "grade strictly against the rubric" in ctx
    # Whitespace-only entries count as absent too.
    assert _context({"HD": "   ", "DI": ""}).count("none provided") == 1


def test_full_system_text_builds_for_every_subset():
    subsets = [{}, {"HD": "x" * 50}, {"AVG": "y" * 50, "PASS": "z" * 50}]
    for samples in subsets:
        text = build_grading_system_text("task", "rubric", _COMPONENTS, samples)
        assert "# CALIBRATION ANCHORS" in text


# --- Engine grades a cohort with fewer than four anchors ----------------------

def test_grade_batch_runs_with_one_sample():
    from src.document_parser import ParsedDocument
    from src.grading_engine import GradingSession, ProviderSpec, grade_batch_with_fallback
    from tests.test_fallback import FakeGrader, _builder

    session = GradingSession(
        task_text="task", rubric_text="rubric", components=_COMPONENTS,
        reference_samples={"HD": "only the HD sample"}, name="partial",
    )
    docs = [ParsedDocument(f"d{i}.txt", f"report {i}") for i in range(3)]
    result = grade_batch_with_fallback(
        [ProviderSpec("p", label="p")], session, docs,
        builder=_builder({"p": FakeGrader("p", score=70)}),
    )
    assert len(result.reports) == 3
    assert result.errors == []


# --- Multi-file sample slots --------------------------------------------------

def test_uploads_text_concatenates_and_labels_files(monkeypatch):
    import app
    from src.document_parser import ParsedDocument

    class FakeUpload:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        app, "parse_upload",
        lambda up: [ParsedDocument(up.name, f"text of {up.name}")],
    )
    one = app._uploads_text([FakeUpload("essay.docx")])
    assert one == "text of essay.docx"          # single file: no label noise

    merged = app._uploads_text([FakeUpload("essay.docx"), FakeUpload("calcs.xlsx")])
    assert "=== File: essay.docx ===" in merged
    assert "=== File: calcs.xlsx ===" in merged
    assert "text of calcs.xlsx" in merged

    assert app._uploads_text([]) == ""


# --- Grade-safe wording: the persona must never teach praise-rating ----------

def test_persona_uses_describe_not_rate_wording():
    from src.prompts import GRADING_SYSTEM_PROMPT, build_grading_user_message

    assert "GRADE-SAFE WORDING" in GRADING_SYSTEM_PROMPT
    assert "ACKNOWLEDGE BY DESCRIBING, NOT RATING" in GRADING_SYSTEM_PROMPT
    # The old tone examples taught exactly the phrasing students appeal with.
    assert "Good work overall" not in GRADING_SYSTEM_PROMPT
    assert "Very good identification" not in GRADING_SYSTEM_PROMPT
    # Praise vocabulary is explicitly gated to top-of-range components
    # (normalise whitespace: the phrase wraps across lines in the prompt).
    flat = " ".join(GRADING_SYSTEM_PROMPT.split())
    assert "at the very top of its range" in flat
    # And the per-request reminder reinforces it (persona rules alone proved
    # unreliable until repeated per request - see the Q-accounting history).
    msg = build_grading_user_message("s1_report.docx", "body")
    assert "not " in msg and "rating it" in msg
