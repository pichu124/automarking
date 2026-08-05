"""Tests: per-student file grouping + marker-only materials in context."""
from __future__ import annotations

from src import config
from src.document_parser import ParsedDocument, group_by_student
from src.grading_engine import GradingSession
from src.models import RubricComponent
from src.prompts import build_grading_system_text


# --- Grouping --------------------------------------------------------------

def test_groups_word_and_excel_by_student_id():
    docs = [
        ParsedDocument("ASM4_BAFI3230_s4180007 (1)-1.docx", "word text"),
        ParsedDocument("Assignment 4_BAFI3230_s4099915_TranHaLinh.docx", "other student"),
        ParsedDocument("s4180007_workbook.xlsx", "excel rows"),
    ]
    out = group_by_student(docs)
    assert len(out) == 2
    merged = next(d for d in out if "s4180007" in d.name)
    # Word file is primary: labels the group and its text comes first.
    assert merged.name.startswith("ASM4_BAFI3230_s4180007")
    assert "[+1 file(s)]" in merged.name
    assert merged.text.index("word text") < merged.text.index("excel rows")
    assert "=== File: s4180007_workbook.xlsx ===" in merged.text


def test_groups_by_matching_stem_without_id():
    docs = [
        ParsedDocument("Nguyen Van A - Report.docx", "essay"),
        ParsedDocument("Nguyen Van A - Report.xlsx", "numbers"),
        ParsedDocument("Tran B.docx", "solo essay"),
    ]
    out = group_by_student(docs)
    assert len(out) == 2
    merged = next(d for d in out if "Nguyen Van A" in d.name)
    assert "essay" in merged.text and "numbers" in merged.text


def test_grouping_disabled_keeps_files_separate():
    docs = [
        ParsedDocument("s4180007.docx", "a"),
        ParsedDocument("s4180007.xlsx", "b"),
    ]
    assert len(group_by_student(docs, enabled=False)) == 2


def test_singletons_unchanged():
    docs = [ParsedDocument("s4111111_only.docx", "a")]
    out = group_by_student(docs)
    assert out[0].name == "s4111111_only.docx"
    assert out[0].text == "a"


# --- Marker-only materials --------------------------------------------------

def _components():
    return [RubricComponent(name="C", weight=100.0, description="")]


def test_marker_materials_included_with_framing():
    text = build_grading_system_text(
        "task", "rubric", _components(), {},
        marker_materials="Q2b tangency weights: w1=1.0, w2=0.0",
    )
    assert "MARKER-ONLY MATERIALS" in text
    assert "error-carried-forward" in text.lower() or "ONCE at its source" in text
    assert "never state the official correct" in text.lower() or "never state the official" in text.lower()
    assert "Q2b tangency weights" in text


def test_no_marker_section_when_empty():
    text = build_grading_system_text("task", "rubric", _components(), {})
    assert "MARKER-ONLY MATERIALS" not in text


def test_marker_materials_capped():
    huge = "K" * (config.MAX_MARKER_CHARS + 50_000)
    text = build_grading_system_text(
        "task", "rubric", _components(), {}, marker_materials=huge
    )
    assert "truncated" in text
    assert len(text) < config.MAX_MARKER_CHARS + 20_000


def test_session_carries_marker_materials_to_grader():
    session = GradingSession(
        task_text="t", rubric_text="r", components=_components(),
        reference_samples={}, marker_materials="SECRET KEY CONTENT",
    )
    captured = {}

    class Spy:
        label = "spy"

        def grade_report(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

    from src.grading_engine import grade_batch
    from src.document_parser import ParsedDocument as PD
    grade_batch(Spy(), session, [PD("x.docx", "body")])
    assert captured.get("marker_materials") == "SECRET KEY CONTENT"
