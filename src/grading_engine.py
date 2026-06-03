"""Orchestrates the Two-Pass grading workflow.

    Pass 1  Initial Grading  -> grade every report independently (LLM)
    Pass 2  Curve Calibration -> map raw totals onto the target distribution
            Final Adjustment   -> rescale each report's component scores to its
                                   calibrated total

The engine is UI-agnostic; ``app.py`` drives it and renders progress. A
``progress_callback(done, total, label)`` can be passed to report progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .calibration import CalibrationConfig, CalibrationReport, calibrate_scores, rescale_components
from .document_parser import ParsedDocument
from .llm_client import GradingLLM
from .models import CalibratedReport, ComponentGrade, GradingResult, RubricComponent

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class GradingSession:
    """All context for one grading session (one assignment)."""

    task_text: str
    rubric_text: str
    components: List[RubricComponent]
    reference_samples: dict[str, str]  # keys: HD, DI, AVG, PASS
    name: str = "Untitled session"


@dataclass
class BatchResult:
    reports: List[CalibratedReport] = field(default_factory=list)
    calibration: Optional[CalibrationReport] = None
    errors: List[str] = field(default_factory=list)


def grade_batch(
    llm: GradingLLM,
    session: GradingSession,
    documents: List[ParsedDocument],
    calibration_config: CalibrationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchResult:
    """Run both passes over a batch of student documents."""
    result = BatchResult()

    # --- Pass 1: independent grading -------------------------------------
    raw_results: List[tuple[ParsedDocument, GradingResult]] = []
    total = len(documents)
    for idx, doc in enumerate(documents):
        if progress_callback:
            progress_callback(idx, total, f"Grading {doc.name}")
        try:
            graded = llm.grade_report(
                task_text=session.task_text,
                rubric_text=session.rubric_text,
                components=session.components,
                reference_samples=session.reference_samples,
                source_name=doc.name,
                report_text=doc.text,
            )
            raw_results.append((doc, graded))
        except Exception as exc:  # keep grading the rest of the batch
            result.errors.append(f"{doc.name}: {exc}")

    if progress_callback:
        progress_callback(total, total, "Calibrating cohort")

    if not raw_results:
        return result

    # --- Pass 2: cohort calibration --------------------------------------
    raw_totals = [_clamp_total(g.total) for _, g in raw_results]
    final_totals, calib = calibrate_scores(raw_totals, calibration_config)
    result.calibration = calib

    # --- Final adjustment: rescale components to calibrated totals -------
    for (doc, graded), raw_total, final_total in zip(raw_results, raw_totals, final_totals):
        max_scores = [c.max_score for c in graded.components]
        scores = [c.score for c in graded.components]
        new_scores = rescale_components(scores, max_scores, raw_total, final_total)
        components = [
            ComponentGrade(
                name=c.name,
                score=new_scores[i],
                max_score=c.max_score,
                notes=c.notes,
            )
            for i, c in enumerate(graded.components)
        ]
        result.reports.append(
            CalibratedReport(
                student_identifier=graded.student_identifier or doc.name,
                source_name=doc.name,
                components=components,
                raw_total=round(raw_total, 1),
                final_total=round(final_total, 1),
            )
        )

    # Present highest-scoring first.
    result.reports.sort(key=lambda r: r.final_total, reverse=True)
    return result


def _clamp_total(total: float) -> float:
    return max(0.0, min(100.0, float(total)))
