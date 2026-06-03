"""Orchestrates the Two-Pass grading workflow, with consistent provider fallback.

    Pass 1  Initial Grading  -> grade every report independently (LLM)
    Pass 2  Curve Calibration -> map raw totals onto the target distribution
            Final Adjustment   -> rescale each report's component scores to its
                                   calibrated total

# Consistency-preserving fallback

The calibration in Pass 2 only makes sense if every report in the cohort was
graded by the *same* model — different models have different score baselines, so
a half-Claude / half-Llama cohort cannot be calibrated on one scale.

So fallback works as follows:

  * Transient rate-limits / 5xx are absorbed *inside* a provider by the SDK's
    own exponential backoff (``max_retries``). Most 429s never cause a switch.
  * Only a persistent provider-health failure (rate-limit after retries, auth
    error, server error, connection error) triggers a switch — and switching
    means **re-grading the whole cohort from scratch** with the next provider.
    Partial results from the failed provider are discarded.
  * Per-report content errors (e.g. an unparseable document, invalid JSON for
    one report) do NOT switch provider; they are recorded and skipped.

The net guarantee: the final cohort is always graded end-to-end by a single
provider. The result records which one, and whether a fallback happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from .calibration import CalibrationConfig, CalibrationReport, calibrate_scores, rescale_components
from .document_parser import ParsedDocument
from .models import CalibratedReport, ComponentGrade, GradingResult, RubricComponent
from .providers import build_provider

ProgressCallback = Callable[[int, int, str], None]


class Grader(Protocol):
    """Any grading back-end (see ``src/providers.py``)."""

    def structure_rubric(self, rubric_text: str) -> "StructuredRubric": ...  # noqa: F821

    def grade_report(
        self,
        *,
        task_text: str,
        rubric_text: str,
        components: List[RubricComponent],
        reference_samples: dict,
        source_name: str,
        report_text: str,
    ) -> GradingResult: ...


@dataclass
class ProviderSpec:
    """A provider the engine may grade with, in fallback priority order."""

    provider_key: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    effort: str = "high"
    temperature: float = 0.2
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.provider_key


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
    notes: List[str] = field(default_factory=list)        # fallback/audit trail
    provider_label: str = ""                              # who graded the cohort
    fell_back_from: List[str] = field(default_factory=list)


# --- Provider-health classification ---------------------------------------

# 413 (request too large) is included so an oversized request falls back to a
# bigger-context provider rather than failing report-by-report.
_HEALTH_STATUS = {401, 403, 408, 409, 413, 429, 500, 502, 503, 504, 529}
_HEALTH_NAME_TOKENS = (
    "ratelimit", "overloaded", "timeout", "connection", "internalservererror",
    "serviceunavailable", "apistatuserror", "apiconnectionerror",
    "authentication", "permissiondenied",
)
# Message fragments that mean "this provider can't serve the batch" — a bad
# API key, exhausted quota, or a too-large request. Catching these lets the
# engine fail fast and switch instead of repeating the same error per report.
_HEALTH_MESSAGE_TOKENS = (
    "api key", "api_key_invalid", "invalid api key", "invalid_api_key",
    "unauthorized", "permission denied", "quota", "insufficient_quota",
    "rate limit", "rate_limit", "request too large", "request_too_large",
    "tokens per minute", "reduce your message size", "billing",
)


class ProviderUnavailable(Exception):
    """Raised when a provider can't reliably serve the batch (-> fall back)."""

    def __init__(self, label: str, reason: str):
        super().__init__(f"{label} unavailable: {reason}")
        self.label = label
        self.reason = reason


def _is_provider_unhealthy(exc: Exception) -> bool:
    """True for errors that mean *the provider* is the problem (not the report).

    Rate-limits, auth/key failures, quota, oversized-request, server and
    connection errors qualify; a plain validation / JSON-parse error does not
    (that's a per-report content issue).
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and status in _HEALTH_STATUS:
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    if any(tok in blob for tok in _HEALTH_NAME_TOKENS):
        return True
    return any(tok in blob for tok in _HEALTH_MESSAGE_TOKENS)


# --- Pass 1 / Pass 2 helpers ----------------------------------------------

def _grade_pass1(
    grader: Grader,
    session: GradingSession,
    documents: Sequence[ParsedDocument],
    progress_callback: ProgressCallback | None,
    raise_on_unhealthy: bool,
) -> tuple[List[tuple[ParsedDocument, GradingResult]], List[str]]:
    """Grade every document with one grader.

    Raises ``ProviderUnavailable`` (when ``raise_on_unhealthy``) on the first
    provider-health failure, so the caller can discard partial work and switch.
    """
    raw_results: List[tuple[ParsedDocument, GradingResult]] = []
    errors: List[str] = []
    total = len(documents)
    for idx, doc in enumerate(documents):
        if progress_callback:
            progress_callback(idx, total, f"Grading {doc.name}")
        try:
            graded = grader.grade_report(
                task_text=session.task_text,
                rubric_text=session.rubric_text,
                components=session.components,
                reference_samples=session.reference_samples,
                source_name=doc.name,
                report_text=doc.text,
            )
            raw_results.append((doc, graded))
        except Exception as exc:
            if raise_on_unhealthy and _is_provider_unhealthy(exc):
                raise ProviderUnavailable(_label_for(grader), str(exc)) from exc
            errors.append(f"{doc.name}: {exc}")
    if progress_callback:
        progress_callback(total, total, "Calibrating cohort")
    return raw_results, errors


def _finalize(
    raw_results: Sequence[tuple[ParsedDocument, GradingResult]],
    calibration_config: CalibrationConfig | None,
) -> tuple[List[CalibratedReport], CalibrationReport]:
    """Pass 2 + final adjustment for a completed Pass 1."""
    raw_totals = [_clamp_total(g.total) for _, g in raw_results]
    final_totals, calib = calibrate_scores(raw_totals, calibration_config)

    reports: List[CalibratedReport] = []
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
        reports.append(
            CalibratedReport(
                student_identifier=graded.student_identifier or doc.name,
                source_name=doc.name,
                components=components,
                raw_total=round(raw_total, 1),
                final_total=round(final_total, 1),
            )
        )
    reports.sort(key=lambda r: r.final_total, reverse=True)
    return reports, calib


def _label_for(grader: Grader) -> str:
    return getattr(grader, "label", type(grader).__name__)


# --- Public API: single provider ------------------------------------------

def grade_batch(
    grader: Grader,
    session: GradingSession,
    documents: List[ParsedDocument],
    calibration_config: CalibrationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchResult:
    """Run both passes with one provider (no fallback). Per-report errors are
    recorded, never raised."""
    result = BatchResult()
    raw_results, result.errors = _grade_pass1(
        grader, session, documents, progress_callback, raise_on_unhealthy=False
    )
    if not raw_results:
        return result
    result.reports, result.calibration = _finalize(raw_results, calibration_config)
    return result


# --- Public API: ordered fallback -----------------------------------------

GraderBuilder = Callable[[ProviderSpec], Grader]


def _default_builder(spec: ProviderSpec) -> Grader:
    grader = build_provider(
        spec.provider_key,
        model=spec.model,
        api_key=spec.api_key,
        base_url=spec.base_url,
        effort=spec.effort,
        temperature=spec.temperature,
    )
    # Tag the grader so logs/results can name the provider that did the work.
    try:
        grader.label = spec.display  # type: ignore[attr-defined]
    except Exception:
        pass
    return grader


def grade_batch_with_fallback(
    specs: Sequence[ProviderSpec],
    session: GradingSession,
    documents: List[ParsedDocument],
    calibration_config: CalibrationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    builder: GraderBuilder = _default_builder,
) -> BatchResult:
    """Grade the cohort, trying each provider in order until one serves it whole.

    Switching providers re-grades the ENTIRE cohort with the new provider, so
    the final result is always internally consistent (one grader, one scale).
    """
    result = BatchResult()
    attempted: List[str] = []

    for index, spec in enumerate(specs):
        label = spec.display
        try:
            grader = builder(spec)
        except Exception as exc:
            result.notes.append(f"Skipped {label}: could not initialise ({exc}).")
            continue

        def cb(done: int, total: int, msg: str, _label=label, _idx=index) -> None:
            if progress_callback is None:
                return
            prefix = "" if _idx == 0 else f"[fallback → {_label}] "
            progress_callback(done, total, prefix + msg)

        try:
            raw_results, per_errors = _grade_pass1(
                grader, session, documents, cb, raise_on_unhealthy=True
            )
        except ProviderUnavailable as exc:
            # Discard partial work — the next provider re-grades from scratch.
            attempted.append(label)
            result.notes.append(
                f"{label} unavailable ({exc.reason}); re-grading the whole cohort "
                f"with the next provider to keep scores consistent."
            )
            continue

        if not raw_results:
            attempted.append(label)
            result.errors.extend(per_errors)
            result.notes.append(f"{label} produced no usable grades; trying next provider.")
            continue

        # Success: this provider graded the whole cohort.
        result.errors = per_errors
        result.reports, result.calibration = _finalize(raw_results, calibration_config)
        result.provider_label = label
        result.fell_back_from = attempted
        return result

    result.errors.append(
        "All configured providers failed: " + ", ".join(attempted or ["(none usable)"])
    )
    return result


def structure_rubric_with_fallback(
    specs: Sequence[ProviderSpec],
    rubric_text: str,
    builder: GraderBuilder = _default_builder,
) -> tuple["StructuredRubric", str]:  # noqa: F821
    """Structure the rubric, trying each provider until one succeeds.

    Returns (rubric, provider_label). Rubric structuring only extracts component
    names/weights, so the provider used here does not affect grading
    consistency — any healthy provider will do.
    """
    last_error: Optional[Exception] = None
    for spec in specs:
        try:
            grader = builder(spec)
            return grader.structure_rubric(rubric_text), spec.display
        except Exception as exc:  # try the next provider
            last_error = exc
    raise RuntimeError(f"Rubric structuring failed on all providers: {last_error}")


def _clamp_total(total: float) -> float:
    return max(0.0, min(100.0, float(total)))
