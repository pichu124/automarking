"""Policy regressions: the rubric's weights bound each component (not the
model's echoed maxima), and cost is reported honestly — a thinking range up
front, the real token bill after a run. No network: every client is a fake."""
from __future__ import annotations

import pytest

from src.document_parser import ParsedDocument
from src.grading_engine import GradingSession, grade_batch
from src.models import ComponentGrade, GradingResult, RubricComponent

# A "20 marks + 10 marks" rubric after weight normalisation to 100.
RUBRIC = [
    RubricComponent(name="Analysis", weight=66.67, description=""),
    RubricComponent(name="Writing", weight=33.33, description=""),
]


@pytest.fixture(autouse=True)
def _dummy_keys(monkeypatch):
    # The repo's .env holds REAL keys; nothing here may reach a provider.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(var, "test-dummy")


def _session(components=RUBRIC) -> GradingSession:
    return GradingSession(
        task_text="task", rubric_text="rubric", components=components, reference_samples={}
    )


def _result(*components: tuple[str, float, float]) -> GradingResult:
    return GradingResult(
        student_identifier="s",
        components=[ComponentGrade(name=n, score=s, max_score=m, notes=["n"])
                    for n, s, m in components],
        total=sum(s for _, s, _ in components),
    )


class _Scripted:
    """Grader returning a preset GradingResult per document name."""

    label = "fake"

    def __init__(self, by_name: dict[str, GradingResult]):
        self.by_name = by_name

    def grade_report(self, *, source_name, **_):
        return self.by_name[source_name]


def _grade(by_name, components=RUBRIC):
    docs = [ParsedDocument(name, "body") for name in by_name]
    return grade_batch(_Scripted(by_name), _session(components), docs)


def _components(res, source_name):
    graded = dict(res.raw_grades)[source_name]
    return {c.name: (round(c.score, 2), c.max_score) for c in graded.components}


# --- Change 1: rubric weights are each component's maximum -------------------

def test_raw_rubric_marks_are_rescaled_to_weights_instead_of_capping_the_cohort():
    # The model echoes "out of 20 / out of 10": before the fix every total
    # was capped at 30, so the whole cohort failed.
    res = _grade({
        f"s{i}.docx": _result(("Analysis", 20 - 2 * i, 20), ("Writing", 10 - i, 10))
        for i in range(5)
    })
    assert res.errors == []
    assert _components(res, "s1.docx") == {"Analysis": (60.0, 66.67), "Writing": (30.0, 33.33)}
    raw = {r.source_name: r.raw_total for r in res.reports}
    assert raw == {"s0.docx": 100.0, "s1.docx": 90.0, "s2.docx": 80.0,
                   "s3.docx": 70.0, "s4.docx": 60.0}
    assert 65.0 <= res.calibration.final_mean <= 66.0
    for r in res.reports:
        assert [c.max_score for c in r.components] == [66.67, 33.33]
        assert sum(c.score for c in r.components) == pytest.approx(r.final_total, abs=0.11)
    fixes = [n for n in res.notes if "rubric weight" in n]
    assert len(fixes) == 5
    assert fixes[0].startswith("s0.docx: ") and "'Analysis' 20 → 66.67" in fixes[0]


def test_reordered_components_are_paired_by_name():
    res = _grade({"a.docx": _result(("writing ", 4, 10), ("ANALYSIS", 15, 20))})
    assert _components(res, "a.docx") == {
        "writing ": (13.33, 33.33), "ANALYSIS": (50.0, 66.67),
    }


def test_unmatched_names_are_paired_by_position():
    res = _grade({"a.docx": _result(("Part 1", 15, 20), ("Part 2", 4, 10))})
    assert _components(res, "a.docx") == {
        "Part 1": (50.0, 66.67), "Part 2": (13.33, 33.33),
    }


def test_non_positive_model_max_clamps_the_score_into_the_weight():
    res = _grade({"a.docx": _result(("Analysis", 50, 0), ("Writing", 20, -5))})
    assert _components(res, "a.docx") == {
        "Analysis": (50.0, 66.67), "Writing": (20.0, 33.33),
    }


def test_maxima_within_one_percent_of_the_weight_are_left_alone():
    res = _grade({"a.docx": _result(("Analysis", 40, 66.5), ("Writing", 20, 33.6))})
    assert _components(res, "a.docx") == {
        "Analysis": (40.0, 66.5), "Writing": (20.0, 33.6),
    }
    assert res.notes == []


def test_component_count_mismatch_keeps_model_maxima_and_notes_it():
    res = _grade({"a.docx": _result(("A", 10, 20), ("B", 5, 10), ("C", 5, 10))})
    assert _components(res, "a.docx") == {"A": (10.0, 20), "B": (5.0, 10), "C": (5.0, 10)}
    assert len(res.notes) == 1
    assert res.notes[0].startswith("a.docx: ") and "3 components" in res.notes[0]
