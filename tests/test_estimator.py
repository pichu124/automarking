"""Tests for the pre-flight cost/time estimator (no network)."""
from __future__ import annotations

from src.config import ModelPricing
from src.document_parser import ParsedDocument
from src.estimator import estimate_batch, estimate_tokens


def _est(pricing, n_reports=10, report_chars=4000, n_components=4):
    return estimate_batch(
        provider_key="x",
        provider_label="X",
        model="m",
        static_context_text="C" * 8000,        # ~2000 tokens of shared context
        report_texts=["R" * report_chars] * n_reports,
        n_components=n_components,
        pricing=pricing,
    )


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 4000) == 1000


def test_free_tier_label_zero_cost():
    pricing = ModelPricing(0.0, 0.0, free_tier=True)
    est = _est(pricing)
    assert est.cost_usd == 0.0
    assert "Free" in est.cost_label


def test_caching_reduces_billed_input_vs_no_cache():
    # Same prices, only difference is cache support -> Claude bills the static
    # prefix once (write) + cheap reads, so fewer effective input tokens.
    cached = ModelPricing(5.0, 25.0, supports_cache=True)
    uncached = ModelPricing(5.0, 25.0, supports_cache=False)
    e_cache = _est(cached, n_reports=20)
    e_plain = _est(uncached, n_reports=20)
    assert e_cache.input_tokens_billed < e_plain.input_tokens_billed
    assert e_cache.cost_usd < e_plain.cost_usd


def test_cost_scales_with_reports():
    pricing = ModelPricing(1.0, 4.0)
    small = _est(pricing, n_reports=5)
    big = _est(pricing, n_reports=50)
    assert big.cost_usd > small.cost_usd
    assert big.output_tokens == 10 * small.output_tokens


def test_time_label_formats():
    est = _est(ModelPricing(1.0, 4.0, seconds_per_report=8.0), n_reports=10)
    assert "≈" in est.time_label
    assert est.seconds_high > est.seconds_low
