"""Pydantic models used for structured LLM output and for moving graded data
between the engine, the calibrator and the .docx exporter.

These schemas double as the structured-output contract sent to Claude via
``messages.parse`` / ``output_config.format``.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# --- Rubric structuring ----------------------------------------------------

class RubricComponent(BaseModel):
    """One graded line of the rubric."""

    name: str = Field(description="Exact name of the rubric component/criterion.")
    weight: float = Field(
        description="Maximum points for this component on a 0-100 scale. "
        "Across all components these must sum to 100."
    )
    description: str = Field(
        description="What this component assesses, paraphrased from the rubric."
    )


class StructuredRubric(BaseModel):
    """The uploaded rubric parsed into ordered, weighted components."""

    components: List[RubricComponent]


# --- Per-report grading output --------------------------------------------

class ComponentGrade(BaseModel):
    """The grade and feedback for a single rubric component on one report."""

    name: str = Field(description="Rubric component name (must match the rubric).")
    score: float = Field(description="Points awarded, between 0 and max_score.")
    max_score: float = Field(description="Maximum points for this component.")
    notes: List[str] = Field(
        description="1-3 feedback notes addressed to the student in second "
        "person, per the persona in the system prompt: complete sentences "
        "citing specific content from their submission, kept as brief as the "
        "point allows. A genuinely strong component may need only one short "
        "approving note."
    )


class GradingResult(BaseModel):
    """A fully graded report (Pass 1, before calibration)."""

    student_identifier: str = Field(
        description="Student name or ID extracted from the report, "
        "or the source filename if none is present."
    )
    components: List[ComponentGrade]
    total: float = Field(
        description="Sum of component scores, 0-100. Equals the percentage grade."
    )


# --- Internal representation (post-calibration) ----------------------------

class CalibratedReport(BaseModel):
    """A graded report after the calibration curve has been applied.

    ``raw_total`` is the Pass-1 score; ``final_total`` is the calibrated grade.
    Component scores are rescaled to sum to ``final_total``.
    """

    student_identifier: str
    source_name: str
    components: List[ComponentGrade]
    raw_total: float
    final_total: float

    @property
    def band(self) -> str:
        t = self.final_total
        if t >= 80:
            return "High Distinction"
        if t >= 70:
            return "Distinction"
        if t >= 60:
            return "Credit"
        if t >= 50:
            return "Pass"
        return "Fail"
