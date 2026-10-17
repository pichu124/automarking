"""AppTest regressions for UI state bugs: grouping edits silently dropped,
a failed re-run clobbering good results, an API key leaking to another
provider, stale rubric/batch results shown as current, and lost runs.

Runs the real app.py headlessly; every LLM call is replaced by a fake.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

import src.grading_engine as ge  # noqa: E402
from src.models import (  # noqa: E402
    ComponentGrade,
    GradingResult,
    RubricComponent,
    StructuredRubric,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
COMPONENTS = [RubricComponent(name="Analysis", weight=100, description="d")]
BATCH = ["alpha.txt", "beta.txt", "gamma.txt"]
_REAL_GRADE = ge.grade_batch_with_fallback


class _Grades70:
    def grade_report(self, **kw):
        return GradingResult(
            student_identifier=kw["source_name"],
            components=[ComponentGrade(name="Analysis", score=70, max_score=100, notes=["ok"])],
            total=70,
        )


class _Unauthorized(Exception):
    status_code = 401


class _RejectsKey:
    def grade_report(self, **kw):
        raise _Unauthorized("Error code: 401 - invalid x-api-key")


def _use_grader(monkeypatch, grader) -> None:
    monkeypatch.setattr(
        ge, "grade_batch_with_fallback",
        lambda specs, *a, **k: _REAL_GRADE(specs, *a, builder=lambda spec: grader, **k),
    )


def _app(**state):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _upload_batch(at, names: list[str]) -> None:
    uploader = [u for u in at.file_uploader if "Drop all reports" in u.label][0]
    uploader.set_value(
        [(n, f"The report in {n}. ".encode() * 100, "text/plain") for n in names]
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def _ready_app(names=BATCH):
    at = _app(task_text="Write an essay.", rubric_text="Analysis: 100", components=COMPONENTS)
    _upload_batch(at, names)
    return at


def _click(at, label: str) -> None:
    [b for b in at.button if label in b.label][0].click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def _warnings(at) -> str:
    return " | ".join(w.value for w in at.warning)


# --- 1. Adjust-grouping editor ----------------------------------------------

def _group_editor_id(node):
    proto = getattr(node, "proto", None)
    if getattr(proto, "editing_mode", 0):  # READ_ONLY dataframes are 0
        return proto.id
    for child in getattr(node, "children", {}).values():
        found = _group_editor_id(child)
        if found:
            return found
    return None


def _edit_groups(at, edited_rows: dict) -> None:
    """Send the editor's state as the browser does: CUMULATIVE row deltas
    against the data the editor was rendered with. AppTest has no public
    data_editor API, hence the raw widget state."""
    states = at._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id = _group_editor_id(at._tree)
    widget.string_value = json.dumps(
        {"edited_rows": edited_rows, "added_rows": [], "deleted_rows": []}
    )
    at._run(widget_state=states)
    assert not at.exception, [str(e) for e in at.exception]


def test_three_consecutive_grouping_edits_all_persist():
    at = _ready_app()
    edits: dict = {}
    for row in range(3):
        edits[str(row)] = {"Group": "T7"}
        _edit_groups(at, edits)
        # Checked after EVERY edit: the bug dropped edit 2 on its own run.
        assert at.session_state["manual_groups"] == {n: "T7" for n in BATCH[: row + 1]}
    assert len(at.session_state["documents"]) == 1


def test_grouping_toggle_regroups_and_later_edits_persist():
    at = _ready_app(["s4180007_report.txt", "s4180007_data.csv", "other.txt"])
    assert len(at.session_state["documents"]) == 2

    [c for c in at.checkbox if "Group files by student" in c.label][0].set_value(False)
    at.run()
    assert len(at.session_state["documents"]) == 3  # no stale labels pin the old grouping
    assert at.session_state["manual_groups"] == {}

    _edit_groups(at, {"0": {"Group": "T7"}})
    assert at.session_state["manual_groups"] == {"s4180007_report.txt": "T7"}
    _edit_groups(at, {"0": {"Group": "T7"}, "2": {"Group": "T7"}})
    assert at.session_state["manual_groups"] == {
        "s4180007_report.txt": "T7", "other.txt": "T7",
    }
    assert len(at.session_state["documents"]) == 2


# --- 2. A failed re-run must not replace good results -------------------------

def test_failed_rerun_keeps_previous_results_and_reports_the_failure(monkeypatch):
    at = _ready_app()
    _use_grader(monkeypatch, _Grades70())
    _click(at, "Run Two-Pass")
    assert len(at.session_state["batch_result"].reports) == 3

    _use_grader(monkeypatch, _RejectsKey())
    _click(at, "Run Two-Pass")

    assert len(at.session_state["batch_result"].reports) == 3
    errors = " | ".join(e.value for e in at.error)
    assert "invalid x-api-key" in errors
    assert "next provider" not in errors  # there is no next provider


def test_total_failure_on_first_run_is_an_error_not_a_failed_report(monkeypatch):
    at = _ready_app()
    _use_grader(monkeypatch, _RejectsKey())
    _click(at, "Run Two-Pass")

    assert at.session_state["batch_result"] is None
    assert "invalid x-api-key" in " | ".join(e.value for e in at.error)
    assert not [e for e in at.expander if "report(s) failed" in e.label]


# --- 3. API key must not follow a provider switch -----------------------------

def test_api_key_is_not_sent_to_a_different_provider(monkeypatch):
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = _app()
    at.sidebar.selectbox[0].set_value("gemini")
    at.run()
    [t for t in at.sidebar.text_input if t.label == "GEMINI_API_KEY"][0].input("AIza-FAKE")
    at.run()
    assert at.session_state["api_key"] == "AIza-FAKE"

    at.sidebar.selectbox[0].set_value("groq")
    at.run()

    assert at.session_state["api_key"] == ""
    assert "Set GROQ_API_KEY" in " | ".join(w.value for w in at.sidebar.warning)


# --- 4. Stale rubric components / batch results are flagged ------------------

def test_editing_the_rubric_flags_stale_components(monkeypatch):
    monkeypatch.setattr(
        ge, "structure_rubric_with_fallback",
        lambda specs, text: (StructuredRubric(components=COMPONENTS), "Fake"),
    )
    at = _app()

    def rubric_area():
        return [t for t in at.text_area if "Rubric text" in t.label][0]

    rubric_area().input("Analysis: 100")
    at.run()
    _click(at, "Structure rubric")
    assert "Rubric changed" not in _warnings(at)

    rubric_area().input("Analysis: 60\nReferencing: 40")
    at.run()

    assert "Rubric changed" in _warnings(at)
    assert at.session_state["components"] == COMPONENTS  # flagged, not deleted


def test_new_batch_flags_previous_results(monkeypatch):
    _use_grader(monkeypatch, _Grades70())
    at = _ready_app()
    _click(at, "Run Two-Pass")
    assert "previous batch" not in _warnings(at)

    _upload_batch(at, ["delta.txt"])

    assert "previous batch" in _warnings(at)
    assert len(at.session_state["batch_result"].reports) == 3  # flagged, not deleted

