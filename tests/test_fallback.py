"""Tests for consistency-preserving provider fallback.

The critical invariant: whatever happens, the final cohort is graded end-to-end
by a SINGLE provider — never a mix — so the calibration scale is consistent.
"""
from __future__ import annotations

from src.document_parser import ParsedDocument
from src.grading_engine import (
    GradingSession,
    ProviderSpec,
    _is_provider_unhealthy,
    grade_batch_with_fallback,
)
from src.models import ComponentGrade, GradingResult, RubricComponent, StructuredRubric


# --- Fakes -----------------------------------------------------------------

class FakeRateLimit(Exception):
    status_code = 429  # looks like a provider-health failure


class FakeBadContent(Exception):
    pass  # no status, no health token in name -> per-report content error


class FakeGrader:
    def __init__(self, label, score=70.0, unhealthy_on=None, content_err_on=None):
        self.label = label
        self.score = score
        self.unhealthy_on = set(unhealthy_on or [])
        self.content_err_on = set(content_err_on or [])
        self.graded: list[str] = []

    def grade_report(self, *, source_name, **_):
        if source_name in self.unhealthy_on:
            raise FakeRateLimit("429 rate limited")
        if source_name in self.content_err_on:
            raise FakeBadContent("invalid json")
        self.graded.append(source_name)
        return GradingResult(
            student_identifier=source_name,
            components=[ComponentGrade(name="C", score=self.score, max_score=100.0, notes=["x"])],
            total=self.score,
        )

    def structure_rubric(self, rubric_text):
        return StructuredRubric(components=[RubricComponent(name="C", weight=100.0, description="")])


def _builder(mapping):
    def build(spec: ProviderSpec):
        g = mapping.get(spec.provider_key)
        if g is None:
            raise RuntimeError(f"no grader for {spec.provider_key}")
        return g
    return build


def _session():
    return GradingSession(
        task_text="task", rubric_text="rubric",
        components=[RubricComponent(name="C", weight=100.0, description="")],
        reference_samples={}, name="t",
    )


def _docs(n):
    return [ParsedDocument(f"d{i}.txt", f"report {i}") for i in range(n)]


# --- Tests -----------------------------------------------------------------

def test_health_failure_regrades_whole_cohort_with_next_provider():
    primary = FakeGrader("primary", score=80, unhealthy_on={"d2.txt"})
    secondary = FakeGrader("secondary", score=60)
    specs = [ProviderSpec("primary", label="primary"), ProviderSpec("secondary", label="secondary")]

    docs = _docs(5)
    result = grade_batch_with_fallback(
        specs, _session(), docs, builder=_builder({"primary": primary, "secondary": secondary})
    )

    # Consistency: every report graded by the SAME (secondary) provider.
    assert result.provider_label == "secondary"
    assert result.fell_back_from == ["primary"]
    assert len(result.reports) == 5
    assert secondary.graded == ["d0.txt", "d1.txt", "d2.txt", "d3.txt", "d4.txt"]
    # Primary's partial work was discarded (it stopped at the failing doc).
    assert primary.graded == ["d0.txt", "d1.txt"]


def test_recalibrate_reuses_grades_without_regrading():
    import dataclasses
    from src.config import DEFAULT_CALIBRATION
    from src.grading_engine import recalibrate

    graded = FakeGrader("primary", score=80)
    specs = [ProviderSpec("primary", label="primary")]
    docs = _docs(12)
    result = grade_batch_with_fallback(
        specs, _session(), docs, builder=_builder({"primary": graded})
    )
    assert len(result.raw_grades) == 12
    n_graded_first = len(graded.graded)

    # Re-calibrate with a different mean target — no further grade_report calls.
    cfg = dataclasses.replace(DEFAULT_CALIBRATION, target_mean_low=61.5, target_mean_high=62.5)
    recal = recalibrate(result, cfg)
    assert len(graded.graded) == n_graded_first  # LLM not called again
    assert len(recal.reports) == 12
    assert recal.provider_label == "primary"
    assert 61.0 <= recal.calibration.final_mean <= 63.0  # new mean applied


def test_content_error_does_not_switch_provider():
    primary = FakeGrader("primary", content_err_on={"d2.txt"})
    secondary = FakeGrader("secondary")
    specs = [ProviderSpec("primary", label="primary"), ProviderSpec("secondary", label="secondary")]

    docs = _docs(4)
    result = grade_batch_with_fallback(
        specs, _session(), docs, builder=_builder({"primary": primary, "secondary": secondary})
    )

    # Stayed on primary; the bad report is an error, the rest are graded.
    assert result.provider_label == "primary"
    assert result.fell_back_from == []
    assert len(result.reports) == 3
    assert any("d2.txt" in e for e in result.errors)
    assert secondary.graded == []  # fallback never used


def test_immediate_failure_falls_back():
    primary = FakeGrader("primary", unhealthy_on={"d0.txt"})
    secondary = FakeGrader("secondary")
    specs = [ProviderSpec("primary", label="primary"), ProviderSpec("secondary", label="secondary")]

    result = grade_batch_with_fallback(
        specs, _session(), _docs(3), builder=_builder({"primary": primary, "secondary": secondary})
    )
    assert result.provider_label == "secondary"
    assert len(result.reports) == 3


def test_all_providers_fail():
    primary = FakeGrader("primary", unhealthy_on={f"d{i}.txt" for i in range(3)})
    secondary = FakeGrader("secondary", unhealthy_on={f"d{i}.txt" for i in range(3)})
    specs = [ProviderSpec("primary", label="primary"), ProviderSpec("secondary", label="secondary")]

    result = grade_batch_with_fallback(
        specs, _session(), _docs(3), builder=_builder({"primary": primary, "secondary": secondary})
    )
    assert result.reports == []
    assert result.provider_label == ""
    assert any("All configured providers failed" in e for e in result.errors)


def test_last_provider_failure_keeps_partial_grades_and_resume_skips_them():
    cache: dict = {}
    specs = [ProviderSpec("primary", label="primary")]
    flaky = FakeGrader("primary", unhealthy_on={"d3.txt"})
    result = grade_batch_with_fallback(
        specs, _session(), _docs(5), builder=_builder({"primary": flaky}), cache=cache
    )
    # Kept, but never calibrated on a partial cohort.
    assert [name for name, _ in result.raw_grades] == ["d0.txt", "d1.txt", "d2.txt"]
    assert result.pending == 2
    assert result.reports == [] and result.calibration is None

    healthy = FakeGrader("primary")
    resumed = grade_batch_with_fallback(
        specs, _session(), _docs(5), builder=_builder({"primary": healthy}), cache=cache
    )
    assert healthy.graded == ["d3.txt", "d4.txt"]  # finished reports not paid for twice
    assert len(resumed.reports) == 5 and resumed.pending == 0
    assert resumed.calibration is not None


def test_saved_grades_are_never_reused_across_providers_or_changed_inputs():
    cache: dict = {}
    primary_spec = ProviderSpec("primary", label="primary")
    grade_batch_with_fallback(
        [primary_spec], _session(), _docs(5),
        builder=_builder({"primary": FakeGrader("primary", unhealthy_on={"d3.txt"})}),
        cache=cache,
    )

    # Primary dies again; the fallback re-grades the WHOLE cohort itself.
    primary = FakeGrader("primary", unhealthy_on={"d3.txt"})
    secondary = FakeGrader("secondary")
    result = grade_batch_with_fallback(
        [primary_spec, ProviderSpec("secondary", label="secondary")], _session(), _docs(5),
        builder=_builder({"primary": primary, "secondary": secondary}), cache=cache,
    )
    assert primary.graded == []  # d0-d2 came from the cache
    assert secondary.graded == [f"d{i}.txt" for i in range(5)]
    assert result.provider_label == "secondary"

    # Another model, or an edited report, is graded afresh.
    other_model = FakeGrader("primary")
    grade_batch_with_fallback(
        [ProviderSpec("primary", model="other", label="primary")], _session(), _docs(5),
        builder=_builder({"primary": other_model}), cache=cache,
    )
    assert len(other_model.graded) == 5
    edited = FakeGrader("primary")
    docs = _docs(5)
    docs[0] = ParsedDocument("d0.txt", "fixed upload")
    grade_batch_with_fallback(
        [ProviderSpec("primary", model="other", label="primary")], _session(), docs,
        builder=_builder({"primary": edited}), cache=cache,
    )
    assert edited.graded == ["d0.txt"]


def test_summarize_samples_reads_full_and_compresses():
    from src.grading_engine import summarize_samples_with_fallback

    class SummGrader(FakeGrader):
        def __init__(self, label):
            super().__init__(label)
            self.seen_lengths = {}

        def complete_text(self, system, user, max_tokens=1200):
            # The full sample text is in the user message (proves it was read).
            self.seen_lengths[len(user)] = True
            return f"[profile {len(user)} chars in]"

    grader = SummGrader("primary")
    samples = {"HD": "H" * 9000, "DI": "D" * 9000, "AVG": "", "PASS": "P" * 9000}
    profiles, used = summarize_samples_with_fallback(
        [ProviderSpec("primary", label="primary")],
        samples,
        [RubricComponent(name="C", weight=100.0, description="")],
        builder=_builder({"primary": grader}),
    )
    assert used == "primary"
    # Empty AVG skipped; the other three summarised.
    assert set(profiles) == {"HD", "DI", "PASS"}
    assert all(p.startswith("[profile") for p in profiles.values())


def test_is_provider_unhealthy_classification():
    assert _is_provider_unhealthy(FakeRateLimit())
    assert not _is_provider_unhealthy(FakeBadContent())

    class AuthenticationError(Exception):
        pass

    class ValidationError(Exception):
        pass

    assert _is_provider_unhealthy(AuthenticationError())   # name token
    assert not _is_provider_unhealthy(ValidationError())   # content issue
