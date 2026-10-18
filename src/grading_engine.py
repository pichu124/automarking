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
    Partial results from the failed provider never join the next provider's
    cohort; if the LAST provider fails they are returned uncalibrated (with
    ``pending`` set), and with a ``cache`` a later run resumes them.
  * Per-report content errors (e.g. an unparseable document, invalid JSON for
    one report) do NOT switch provider; they are recorded and skipped.

The net guarantee: the final cohort is always graded end-to-end by a single
provider. The result records which one, and whether a fallback happened.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, MutableMapping, Optional, Protocol, Sequence

from .calibration import CalibrationConfig, CalibrationReport, calibrate_scores, rescale_components
from .document_parser import ParsedDocument
from .estimator import CHARS_PER_TOKEN
from .models import CalibratedReport, ComponentGrade, GradingResult, RubricComponent
from . import config
from .prompts import SAMPLE_SUMMARY_SYSTEM, build_sample_summary_user, cap_text
from .providers import ModelOutputError, build_provider

ProgressCallback = Callable[[int, int, str], None]

# Reference-sample slots and their human labels.
SAMPLE_LEVELS = {
    "HD": "High Distinction",
    "DI": "Distinction",
    "AVG": "Average",
    "PASS": "Pass",
}


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
        marker_materials: str = "",
    ) -> GradingResult: ...

    def complete_text(self, system: str, user: str, max_tokens: int = ...) -> str: ...


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
    # Answer key + marking notes, visible to the model but never to students.
    marker_materials: str = ""


@dataclass
class BatchResult:
    reports: List[CalibratedReport] = field(default_factory=list)
    calibration: Optional[CalibrationReport] = None
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)        # fallback/audit trail
    provider_label: str = ""                              # who graded the cohort
    fell_back_from: List[str] = field(default_factory=list)
    # Pass-1 output kept so the cohort can be re-calibrated (new mean/caps/
    # boundary settings) WITHOUT re-calling the LLM — i.e. for free.
    raw_grades: List[tuple] = field(default_factory=list)  # (source_name, GradingResult)
    # Reports still ungraded when the last provider gave out. Non-zero means
    # raw_grades is a partial, UNCALIBRATED cohort — resume, don't publish.
    pending: int = 0


# --- Provider-health classification ---------------------------------------

# 413 (request too large) is included so an oversized request falls back to a
# bigger-context provider rather than failing report-by-report; 404 (model not
# found) is a provider configuration problem, never report-specific.
_HEALTH_STATUS = {401, 403, 404, 408, 409, 413, 429, 500, 502, 503, 504, 529}
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

    def __init__(self, label: str, reason: str, graded: Sequence[tuple] = ()):
        super().__init__(f"{label} unavailable: {reason}")
        self.label = label
        self.reason = reason
        self.graded = list(graded)  # (source_name, GradingResult) done before it failed


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
    # Token matching is unsafe on content-type exceptions: a ValueError
    # (ModelOutputError, JSON parse, pydantic validation) quotes the MODEL'S OWN
    # OUTPUT or the filename, which can innocently contain "authentication"/
    # "timeout"/"quota" when the report or a rubric component is about such a
    # topic — one odd report would then discard the whole cohort. Real
    # provider-health failures from the SDKs are never ValueErrors.
    if isinstance(exc, ValueError):
        return False
    blob = f"{type(exc).__name__} {exc}".lower()
    if any(tok in blob for tok in _HEALTH_NAME_TOKENS):
        return True
    return any(tok in blob for tok in _HEALTH_MESSAGE_TOKENS)


# --- Context-window overflow: retry the one report with a trimmed body ------

# Phrases that mean "this REQUEST exceeds the model's context window" (Anthropic
# and OpenAI-compatible wordings). Distinct from rate-limit style "request too
# large" (Groq TPM), which correctly falls back to a bigger-context provider.
_CONTEXT_OVERFLOW_TOKENS = (
    "prompt is too long",
    "maximum context length",
    "context_length_exceeded",
    "context window",
    "exceeds the maximum number of tokens",
)
_TOKEN_COUNT_RE = re.compile(r"\b\d[\d,]{3,}\b")
# The model's limit sits next to the word "maximum": "maximum context length is
# 128000" (OpenAI/vLLM/DeepSeek), "> 1000000 maximum" (Anthropic), "maximum
# number of tokens allowed (1048576)" (Gemini), "131072 maximum context length"
# (Mistral).
_TOKEN_LIMIT_RE = re.compile(r"maximum\D{0,40}?(\d[\d,]{3,})|(\d[\d,]{3,})\s+maximum")


def _context_overflow_keep_chars(exc: Exception, current_chars: int) -> int | None:
    """How many chars of the report to keep for a retry, or None if ``exc`` is
    not a context-window overflow.

    APIs state the requested and maximum token counts in the message; the
    requested total is the largest number (any breakdown — "124000 in the
    messages, 16000 in the completion" — is smaller). Trim the report by the
    excess at ~4 chars/token plus 15% headroom. Without parseable numbers,
    halve it.
    """
    msg = str(exc).lower()
    if not any(tok in msg for tok in _CONTEXT_OVERFLOW_TOKENS):
        return None
    nums = [int(n.replace(",", "")) for n in _TOKEN_COUNT_RE.findall(msg)]
    limit = _TOKEN_LIMIT_RE.search(msg)
    excess = 0
    if nums and limit:
        excess = max(nums) - int((limit.group(1) or limit.group(2)).replace(",", ""))
    if excess > 0:
        keep = int(current_chars - excess * CHARS_PER_TOKEN * 1.15)
    else:
        keep = current_chars // 2
    return max(2000, min(keep, current_chars - 1))


# --- Resume cache keys -----------------------------------------------------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _run_key(session: GradingSession, spec: ProviderSpec) -> str:
    """Fingerprint of everything that shapes a grade apart from the report.

    A saved grade is reused only while all of it is unchanged — task, rubric,
    components, anchors, answer key, provider settings and length caps — so a
    resumed cohort stays on one scale. The API key and session name are left
    out: neither changes a grade.
    """
    return _sha1(repr((
        session.task_text, session.rubric_text, session.components,
        sorted(session.reference_samples.items()), session.marker_materials,
        spec.provider_key, spec.model, spec.base_url, spec.effort, spec.temperature,
        config.MAX_REPORT_CHARS, config.MAX_SAMPLE_CHARS,
    )))


def _doc_key(run_key: str, doc: ParsedDocument) -> str:
    return run_key + _sha1(f"{doc.name}\0{doc.text}")


def count_graded(
    specs: Sequence[ProviderSpec],
    session: GradingSession,
    documents: Sequence[ParsedDocument],
    cache: MutableMapping,
) -> tuple[int, int]:
    """(documents with a saved grade, index in ``specs`` of their provider) for
    the first provider holding any for this exact batch and session.

    A resume starts the chain at that index: the providers before it failed
    the run being resumed, and a recovered one would re-grade every report.
    """
    for index, spec in enumerate(specs):
        run_key = _run_key(session, spec)
        done = sum(_doc_key(run_key, d) in cache for d in documents)
        if done:
            return done, index
    return 0, 0


# --- Pass 1 / Pass 2 helpers ----------------------------------------------

def _grade_pass1(
    grader: Grader,
    session: GradingSession,
    documents: Sequence[ParsedDocument],
    progress_callback: ProgressCallback | None,
    raise_on_unhealthy: bool,
    cache: MutableMapping | None = None,
    run_key: str = "",
) -> tuple[List[tuple[str, GradingResult]], List[str], List[str]]:
    """Grade every document with one grader.

    Raises ``ProviderUnavailable`` (when ``raise_on_unhealthy``) on the first
    provider-health failure, carrying the pairs graded so far, so the caller
    can switch provider. Returns ``(pairs, errors, notes)`` where pairs are
    ``(source_name, GradingResult)`` — the raw text is dropped so the result
    can be re-calibrated later without holding the whole batch — and notes
    record non-fatal interventions (e.g. an oversized submission auto-trimmed
    to fit the model's context window). With a ``cache``, documents already in
    it under ``run_key`` are reused (with their notes) instead of graded, and
    each new grade is saved the moment it lands.
    """
    raw_results: List[tuple[str, GradingResult]] = []
    errors: List[str] = []
    notes: List[str] = []
    total = len(documents)
    for idx, doc in enumerate(documents):
        doc_notes_from = len(notes)
        if cache is not None:
            key = _doc_key(run_key, doc)
            if key in cache:
                name, graded, doc_notes = cache[key]
                raw_results.append((name, graded))
                notes.extend(doc_notes)
                continue
        if progress_callback:
            progress_callback(idx, total, f"Grading {doc.name}")
        report_text = doc.text
        try:
            # A submission with a giant spreadsheet dump can exceed the model's
            # context window outright. That is a property of ONE report, so
            # retry it (up to twice) with the body trimmed to fit rather than
            # failing the report or switching provider.
            for attempt in range(3):
                try:
                    graded = grader.grade_report(
                        task_text=session.task_text,
                        rubric_text=session.rubric_text,
                        components=session.components,
                        reference_samples=session.reference_samples,
                        source_name=doc.name,
                        report_text=report_text,
                        marker_materials=session.marker_materials,
                    )
                    break
                except Exception as exc:
                    # Measure what was actually SENT: the provider caps the body
                    # at MAX_REPORT_CHARS, so trimming relative to the raw length
                    # would resend the same capped text.
                    sent = len(report_text)
                    if config.MAX_REPORT_CHARS > 0:
                        sent = min(sent, config.MAX_REPORT_CHARS)
                    keep = _context_overflow_keep_chars(exc, sent)
                    if keep is None or attempt == 2 or keep >= sent:
                        raise
                    report_text = report_text[:keep].rstrip() + (
                        "\n\n[NOTE: the submission was truncated HERE because it "
                        "exceeded the model's context window. Do not treat "
                        "content after this point as missing.]"
                    )
                    notes.append(
                        f"{doc.name}: too large for the model's context window; "
                        f"auto-retried with the first ~{keep:,} characters "
                        "(the tail — usually raw spreadsheet rows — was trimmed)."
                    )
            if not graded.components:
                # Nothing to grade against — recording an error is honest;
                # letting it through would invent a grade from the calibration
                # floor for a report with an EMPTY component table.
                raise ModelOutputError(
                    "model returned no component grades for this report"
                )
            raw_results.append((doc.name, graded))
            if cache is not None:
                # Saved before the next progress update: in Streamlit that is
                # where a click mid-run raises RerunException.
                cache[key] = (doc.name, graded, notes[doc_notes_from:])
        except Exception as exc:
            if raise_on_unhealthy and _is_provider_unhealthy(exc):
                raise ProviderUnavailable(_label_for(grader), str(exc), raw_results) from exc
            errors.append(f"{doc.name}: {exc}")
    if progress_callback:
        progress_callback(total, total, "Calibrating cohort")
    return raw_results, errors, notes


def _finite(x: float) -> float:
    """A float, with NaN/inf zeroed — every comparison against NaN is False,
    so an unguarded clamp would silently turn a NaN score into full marks."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _sane_component_scores(graded: GradingResult) -> List[float]:
    """Component scores clamped into [0, max_score].

    Models occasionally emit impossible values (negative, above the component
    maximum, or NaN/inf); left unchecked these propagate into the exported
    grades.
    """
    return [
        max(0.0, min(max(0.0, _finite(c.max_score)), _finite(c.score)))
        for c in graded.components
    ]


def _sane_raw_total(graded: GradingResult) -> float:
    """The report's raw total, trusting the COMPONENT SUM over `total`.

    The exported table shows component scores and a total; if the model's
    `total` field disagrees with the sum of its own components, the printed
    figures would not add up. The component breakdown is what the marker and
    student can verify, so it wins. `total` is used only when there are no
    components to sum.
    """
    scores = _sane_component_scores(graded)
    if not scores:
        return _clamp_total(graded.total)
    return _clamp_total(sum(scores))


def _finalize(
    raw_results: Sequence[tuple[str, GradingResult]],
    calibration_config: CalibrationConfig | None,
) -> tuple[List[CalibratedReport], CalibrationReport]:
    """Pass 2 + final adjustment for a completed Pass 1.

    Pure calculation — no LLM calls — so it can be re-run for free to apply new
    calibration settings.
    """
    raw_totals = [_sane_raw_total(g) for _, g in raw_results]
    final_totals, calib = calibrate_scores(raw_totals, calibration_config)

    reports: List[CalibratedReport] = []
    for (source_name, graded), raw_total, final_total in zip(raw_results, raw_totals, final_totals):
        max_scores = [max(0.0, _finite(c.max_score)) for c in graded.components]
        scores = _sane_component_scores(graded)
        # A calibrated total can only be distributed across the components that
        # exist; if the model returned a partial rubric (component maxima
        # summing to less than 100) the report is capped at what it can hold,
        # so the printed components always add up to the printed total. (With
        # no components at all — only possible via stored raw_grades from an
        # older version — the cap correctly forces the total to 0.)
        final_total = min(final_total, sum(max_scores))
        # Rescale from the ACTUAL component sum, not the 100-clamped raw total:
        # if a model invented extra components whose maxima sum above 100, the
        # two disagree and the printed table would not add up to the total.
        old_total = sum(scores) if scores else raw_total
        new_scores = rescale_components(scores, max_scores, old_total, final_total)
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
                student_identifier=graded.student_identifier or source_name,
                source_name=source_name,
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
    raw_results, result.errors, pass_notes = _grade_pass1(
        grader, session, documents, progress_callback, raise_on_unhealthy=False
    )
    result.notes.extend(pass_notes)
    if not raw_results:
        return result
    result.raw_grades = list(raw_results)
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
    cache: MutableMapping | None = None,
) -> BatchResult:
    """Grade the cohort, trying each provider in order until one serves it whole.

    Switching providers re-grades the ENTIRE cohort with the new provider, so
    the final result is always internally consistent (one grader, one scale).
    If every provider fails, the first one's partial grades are returned
    uncalibrated with ``pending`` set. ``cache`` (e.g. Streamlit session
    state) gets every grade as it lands and supplies grades the same
    provider already made for the same inputs, so a cut-short run resumes
    without paying for finished reports again.
    """
    result = BatchResult()
    attempted: List[str] = []

    for index, spec in enumerate(specs):
        label = spec.display
        has_next = index < len(specs) - 1
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
            raw_results, per_errors, pass_notes = _grade_pass1(
                grader, session, documents, cb, raise_on_unhealthy=True,
                cache=cache, run_key=_run_key(session, spec),
            )
        except ProviderUnavailable as exc:
            # The next provider re-grades from scratch; this partial work is
            # only kept in case no provider finishes the cohort.
            attempted.append(label)
            result.notes.append(
                f"{label} unavailable ({exc.reason}); "
                + ("re-grading the whole cohort with the next provider to keep "
                   "scores consistent." if has_next else "no fallback provider left.")
            )
            if exc.graded and not result.raw_grades:
                result.raw_grades = exc.graded
                result.provider_label = label
            continue

        if not raw_results:
            attempted.append(label)
            result.errors.extend(per_errors)
            result.notes.append(
                f"{label} produced no usable grades"
                + ("; trying next provider." if has_next else ".")
            )
            continue

        # Success: this provider graded the whole cohort.
        result.errors = per_errors
        result.notes.extend(pass_notes)
        result.raw_grades = list(raw_results)
        result.reports, result.calibration = _finalize(raw_results, calibration_config)
        result.provider_label = label
        result.fell_back_from = attempted
        return result

    result.errors.append(
        "All configured providers failed: " + ", ".join(attempted or ["(none usable)"])
    )
    if result.raw_grades:
        result.pending = len(documents) - len(result.raw_grades)
        result.notes.append(
            f"{result.provider_label} graded {len(result.raw_grades)} of "
            f"{len(documents)} reports before it became unavailable; those grades "
            f"are kept, uncalibrated, so resuming grades only the other "
            f"{result.pending}."
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


def summarize_samples_with_fallback(
    specs: Sequence[ProviderSpec],
    samples: dict,
    components: List[RubricComponent],
    builder: GraderBuilder = _default_builder,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict, str]:
    """Compress each reference sample into a compact anchor profile.

    The model reads each sample IN FULL (one at a time) and distils it into a
    short profile that captures the whole paper's quality — so grading anchors
    survive even on small-context models, where sending the raw samples would
    force lossy truncation. Returns (profiles, provider_label).
    """
    present = [(k, (samples.get(k) or "").strip()) for k in SAMPLE_LEVELS]
    present = [(k, body) for k, body in present if body]
    total = len(present)

    last_error: Optional[Exception] = None
    for spec in specs:
        try:
            grader = builder(spec)
            profiles: dict = {}
            for idx, (key, body) in enumerate(present):
                if progress_callback:
                    progress_callback(idx, total, f"Reading {SAMPLE_LEVELS[key]} sample")
                user = build_sample_summary_user(SAMPLE_LEVELS[key], body, components)
                try:
                    profiles[key] = grader.complete_text(
                        SAMPLE_SUMMARY_SYSTEM, user, config.SAMPLE_SUMMARY_MAX_TOKENS
                    )
                except ModelOutputError:
                    # An empty summary (e.g. a reasoning model spent its whole
                    # budget thinking) must not silently drop this anchor: keep
                    # the sample itself, capped as a raw sample would be, with a
                    # note the professor sees under "Review anchor profiles".
                    profiles[key] = (
                        "[Anchor profile unavailable: the model's summary came "
                        "back empty, so the original sample text is used.]\n\n"
                        + cap_text(body, config.MAX_SAMPLE_CHARS)
                    )
            if progress_callback:
                progress_callback(total, total, "Done")
            return profiles, spec.display
        except Exception as exc:  # try the next provider
            last_error = exc
    raise RuntimeError(f"Sample summarisation failed on all providers: {last_error}")


def recalibrate(
    result: BatchResult,
    calibration_config: CalibrationConfig | None = None,
) -> BatchResult:
    """Re-run Pass 2 (calibration) on an already-graded cohort — no LLM calls.

    Lets the professor tweak the mean, caps or boundary settings and regenerate
    grades for free, reusing the stored Pass-1 output. Returns a fresh
    BatchResult carrying the same provider/audit metadata.
    """
    new = BatchResult(
        errors=list(result.errors),
        notes=list(result.notes),
        provider_label=result.provider_label,
        fell_back_from=list(result.fell_back_from),
        raw_grades=list(result.raw_grades),
    )
    if new.raw_grades:
        new.reports, new.calibration = _finalize(new.raw_grades, calibration_config)
    return new


def _clamp_total(total: float) -> float:
    return max(0.0, min(100.0, _finite(total)))
