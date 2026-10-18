"""Policy regressions: the rubric's weights bound each component (not the
model's echoed maxima), and cost is reported honestly — a thinking range up
front, the real token bill after a run. No network: every client is a fake."""
from __future__ import annotations

import pathlib
from collections import Counter
from types import SimpleNamespace

import pytest

from src.config import PROVIDER_PRICING, ModelPricing
from src.document_parser import ParsedDocument
from src.estimator import THINKING_TOKENS_PER_REPORT, actual_cost_usd, estimate_batch
from src.grading_engine import (
    GradingSession,
    ProviderSpec,
    grade_batch,
    grade_batch_with_fallback,
    recalibrate,
)
from src.models import ComponentGrade, GradingResult, RubricComponent
from src.providers import ModelOutputError, build_provider

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


# --- Change 2a: the estimate is a range that includes Claude's thinking -------

def _est(provider_key, effort="high"):
    return estimate_batch(
        provider_key=provider_key,
        provider_label="X",
        model="m",
        static_context_text="C" * 8000,
        report_texts=["R" * 4000] * 10,
        n_components=4,
        pricing=ModelPricing(5.0, 25.0, supports_cache=True),
        effort=effort,
    )


def test_claude_estimate_is_a_range_that_widens_with_effort():
    plain = _est("x")
    highs = []
    for effort in ("low", "medium", "high", "xhigh", "max"):
        est = _est("anthropic", effort)
        assert est.cost_usd == plain.cost_usd  # low end = the old estimate
        thinking = 10 * THINKING_TOKENS_PER_REPORT[effort]
        assert est.cost_usd_high == pytest.approx(est.cost_usd + thinking * 25 / 1e6, abs=1e-4)
        assert est.cost_label == f"≈ ${est.cost_usd:,.2f}–${est.cost_usd_high:,.2f}"
        highs.append(est.cost_usd_high)
    assert highs == sorted(set(highs))


def test_non_thinking_provider_estimate_is_a_single_figure():
    est = _est("x")
    assert "–" not in est.cost_label
    assert est.cost_label == f"≈ ${est.cost_usd:,.2f}"


# --- Change 2b: real token usage is counted and priced ------------------------

def _anthropic_usage(i, o, cw, cr):
    return SimpleNamespace(input_tokens=i, output_tokens=o,
                           cache_creation_input_tokens=cw, cache_read_input_tokens=cr)


def test_claude_usage_is_counted_from_every_call_even_unparseable_ones():
    provider = build_provider("anthropic")
    parsed = [GradingResult(student_identifier="s", components=[], total=0), None]

    def parse(**_):
        return SimpleNamespace(parsed_output=parsed.pop(0),
                               usage=_anthropic_usage(100, 50, 2000, 0))

    def create(**_):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="profile")],
                               stop_reason="end_turn",
                               usage=_anthropic_usage(10, 5, 0, 2000))

    provider.llm.client = SimpleNamespace(messages=SimpleNamespace(parse=parse, create=create))
    kw = dict(task_text="t", rubric_text="r", components=RUBRIC, reference_samples={},
              source_name="a.docx", report_text="body")
    provider.grade_report(**kw)
    with pytest.raises(ModelOutputError):
        provider.grade_report(**kw)  # billed even though the output is unusable
    provider.complete_text("system", "user")
    assert dict(provider.usage) == {
        "input_tokens": 210, "output_tokens": 105,
        "cache_creation_input_tokens": 4000, "cache_read_input_tokens": 2000,
    }


def test_openai_compatible_usage_is_counted_when_reported():
    provider = build_provider("openai")
    replies = [
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40)),
        SimpleNamespace(),  # some local servers omit usage entirely
    ]

    def create(**_):
        reply = replies.pop(0)
        reply.choices = [SimpleNamespace(
            message=SimpleNamespace(content='{"components": []}'), finish_reason="stop")]
        return reply

    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider.complete_text("system", "user")
    provider.complete_text("system", "user")
    assert dict(provider.usage) == {"input_tokens": 300, "output_tokens": 40}


def test_actual_cost_prices_cache_writes_and_reads_from_the_table():
    p = PROVIDER_PRICING["anthropic"]
    usage = Counter(input_tokens=1_000_000, output_tokens=1_000_000,
                    cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert actual_cost_usd("anthropic", None, usage) == pytest.approx(
        p.input_per_1m * (1 + p.cache_write_mult + p.cache_read_mult) + p.output_per_1m
    )
    assert actual_cost_usd("anthropic", "some-other-claude", usage) is None
    assert actual_cost_usd("custom", "my-model", usage) is None


class _RateLimited(Exception):
    status_code = 429


class _Metered:
    """Grader that bills 1000 in / 200 out per call; optionally rate-limited."""

    def __init__(self, fail: bool):
        self.fail = fail
        self.usage = Counter()

    def grade_report(self, *, source_name, **_):
        self.usage.update(input_tokens=1000, output_tokens=200)
        if self.fail:
            raise _RateLimited("rate limit")
        return _result(("Analysis", 40, 66.67), ("Writing", 20, 33.33))


def test_batch_result_totals_usage_across_fallback_and_prices_each_provider():
    graders = {"anthropic": _Metered(fail=True), "deepseek": _Metered(fail=False)}
    docs = [ParsedDocument("a.docx", "x"), ParsedDocument("b.docx", "y")]
    res = grade_batch_with_fallback(
        [ProviderSpec("anthropic"), ProviderSpec("deepseek")], _session(), docs,
        builder=lambda spec: graders[spec.provider_key],
    )
    assert len(res.reports) == 2
    # The discarded Claude attempt (one call) was still billed.
    assert dict(res.usage) == {"input_tokens": 3000, "output_tokens": 600}
    a, d = PROVIDER_PRICING["anthropic"], PROVIDER_PRICING["deepseek"]
    assert res.cost_usd == pytest.approx(
        (1000 * a.input_per_1m + 200 * a.output_per_1m
         + 2000 * d.input_per_1m + 400 * d.output_per_1m) / 1e6
    )
    again = recalibrate(res)
    assert again.usage == res.usage and again.cost_usd == res.cost_usd


def test_unpriced_model_makes_the_run_cost_unknown():
    res = grade_batch_with_fallback(
        [ProviderSpec("anthropic", model="some-other-claude")], _session(),
        [ParsedDocument("a.docx", "x")], builder=lambda spec: _Metered(fail=False),
    )
    assert dict(res.usage) == {"input_tokens": 1000, "output_tokens": 200}
    assert res.cost_usd is None


# --- App: the range and the real bill are shown -------------------------------

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def test_app_shows_cost_range_before_and_actual_tokens_after_a_run(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    import src.grading_engine as ge

    real = ge.grade_batch_with_fallback
    monkeypatch.setattr(
        ge, "grade_batch_with_fallback",
        lambda specs, *a, **k: real(specs, *a, builder=lambda s: _Metered(fail=False), **k),
    )
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    for key, value in dict(task_text="Write.", rubric_text="r", components=RUBRIC).items():
        at.session_state[key] = value
    at.run()
    uploader = [u for u in at.file_uploader if "Drop all reports" in u.label][0]
    uploader.set_value([(n, b"A report. " * 100, "text/plain") for n in ("a.txt", "b.txt")])
    at.run()
    cost = [m for m in at.metric if m.label == "Est. cost"][0].value
    assert cost.startswith("≈ $") and "–" in cost

    [b for b in at.button if "Run Two-Pass" in b.label][0].click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    line = [c.value for c in at.caption if c.value.startswith("Actual tokens")]
    assert line and "2,000 in" in line[0] and "400 out" in line[0] and "≈ $" in line[0]
