"""The exported numbers must always be internally consistent.

A student reading the grade report adds up the component column and expects it
to equal the printed total. These tests pin that invariant against the kinds of
malformed output models actually produce.
"""
from __future__ import annotations

import random

from src.calibration import rescale_components
from src.grading_engine import _finalize
from src.models import ComponentGrade, GradingResult

RUBRIC = [("AoL5", 15.0), ("Sources", 40.0), ("Assumptions", 35.0), ("Integrity", 10.0)]


def _result(scores, total, ident="S"):
    return GradingResult(
        student_identifier=ident,
        total=total,
        components=[
            ComponentGrade(name=n, score=s, max_score=m, notes=["n"])
            for (n, m), s in zip(RUBRIC, scores)
        ],
    )


def _assert_consistent(report):
    component_sum = sum(c.score for c in report.components)
    assert abs(component_sum - report.final_total) < 0.15, (
        f"components sum to {component_sum} but total says {report.final_total}"
    )
    for c in report.components:
        assert 0 <= c.score <= c.max_score + 0.01, f"{c.name}={c.score}/{c.max_score}"
    assert 0 <= report.final_total <= 100


def test_total_field_disagreeing_with_components_uses_component_sum():
    # Model claims 88 but every component is 0 -> the printed table must add up.
    reports, _ = _finalize([("a.docx", _result([0, 0, 0, 0], total=88.0))], None)
    _assert_consistent(reports[0])


def test_scores_above_max_are_clamped():
    reports, _ = _finalize([("a.docx", _result([22.5, 60, 50, 15], total=150.0))], None)
    _assert_consistent(reports[0])


def test_negative_scores_are_clamped():
    reports, _ = _finalize([("a.docx", _result([-5, -10, -3, -1], total=-20.0))], None)
    _assert_consistent(reports[0])


def test_partial_rubric_total_capped_to_available_maxima():
    # Only one 15-point component exists: the report cannot show 65/100.
    graded = GradingResult(
        student_identifier="S",
        total=90.0,
        components=[ComponentGrade(name="Only", score=9, max_score=15, notes=["n"])],
    )
    reports, _ = _finalize([("a.docx", graded)], None)
    _assert_consistent(reports[0])
    assert reports[0].final_total <= 15.0


def test_cohort_of_malformed_results_stays_consistent():
    random.seed(11)
    cohort = []
    for i in range(25):
        mode = i % 5
        if mode == 0:      # sane
            scores = [10, 28, 24, 7]
            total = sum(scores)
        elif mode == 1:    # total disagrees
            scores = [10, 28, 24, 7]
            total = 95.0
        elif mode == 2:    # over max
            scores = [30, 60, 50, 20]
            total = 160.0
        elif mode == 3:    # negative
            scores = [-2, -5, -1, -1]
            total = -9.0
        else:              # zeros
            scores = [0, 0, 0, 0]
            total = 0.0
        cohort.append((f"s{i}.docx", _result(scores, total, ident=f"S{i}")))
    reports, calib = _finalize(cohort, None)
    assert len(reports) == 25
    for r in reports:
        _assert_consistent(r)
    assert 60 <= calib.final_mean <= 71


def test_rescale_rejects_out_of_range_inputs():
    assert rescale_components([-5.0], [15.0], old_total=-5.0, new_total=10.0) == [10.0]
    assert rescale_components([22.5], [15.0], old_total=22.5, new_total=15.0) == [15.0]
    out = rescale_components([50.0, -5.0], [40.0, 10.0], old_total=45.0, new_total=40.0)
    assert all(0 <= v <= m for v, m in zip(out, [40.0, 10.0]))
