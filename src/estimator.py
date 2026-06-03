"""Pre-flight cost & time estimate for a grading batch.

Token counts are estimated locally (no API call) with a chars-per-token
heuristic — accurate enough to compare providers and avoid surprises before
spending money. The Claude path models prompt caching: the large static context
(persona + task + rubric + reference samples) is written to cache once and read
cheaply for every subsequent report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from .config import DEFAULT_PRICING, PROVIDER_PRICING, ModelPricing
from .document_parser import ParsedDocument
from .prompts import build_grading_system_text

# Rough average across English prose + code; good enough for an estimate.
CHARS_PER_TOKEN = 4.0
# Per-report prompt overhead (the user-message wrapper around the report).
USER_WRAPPER_TOKENS = 80


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text or "") / CHARS_PER_TOKEN)


def _estimate_output_tokens(n_components: int) -> int:
    """Structured grade JSON: identifier + per-component score & 2-3 notes."""
    return 60 + max(1, n_components) * 45


@dataclass
class BatchEstimate:
    provider_key: str
    provider_label: str
    model: str
    n_reports: int
    static_tokens: int          # the cached/shared context, per the system text
    input_tokens_billed: int    # effective input tokens after caching maths
    output_tokens: int
    cost_usd: float
    free_tier: bool
    supports_cache: bool
    seconds_low: float
    seconds_high: float
    # Size of a single grading call (matters for a model's context window, e.g.
    # Ollama's num_ctx) — the static context + one report + the JSON output.
    tokens_per_request_avg: int = 0
    tokens_per_request_max: int = 0

    @property
    def cost_label(self) -> str:
        if self.free_tier and self.cost_usd <= 0.0:
            return "Free (local)"
        if self.free_tier:
            return f"Free tier (else ≈ ${self.cost_usd:,.2f})"
        if self.cost_usd < 0.01:
            return "< $0.01"
        return f"≈ ${self.cost_usd:,.2f}"

    @property
    def time_label(self) -> str:
        return f"≈ {_fmt_time(self.seconds_low)}–{_fmt_time(self.seconds_high)}"


def _fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{round(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{round(minutes)}m"
    return f"{minutes / 60.0:.1f}h"


def estimate_batch(
    *,
    provider_key: str,
    provider_label: str,
    model: str,
    static_context_text: str,
    report_texts: Sequence[str],
    n_components: int,
    pricing: ModelPricing | None = None,
) -> BatchEstimate:
    pricing = pricing or PROVIDER_PRICING.get(provider_key, DEFAULT_PRICING)
    n = len(report_texts)

    static_tokens = estimate_tokens(static_context_text)
    report_tokens = [estimate_tokens(t) for t in report_texts]
    variable_input = sum(report_tokens) + USER_WRAPPER_TOKENS * n
    out_per = _estimate_output_tokens(n_components)
    output_tokens = out_per * n

    if pricing.supports_cache and n > 0:
        # First report writes the static prefix to cache; the rest read it cheap.
        cached_input = (
            static_tokens * pricing.cache_write_mult
            + static_tokens * pricing.cache_read_mult * (n - 1)
        )
        input_tokens_billed = round(cached_input + variable_input)
    else:
        input_tokens_billed = round(static_tokens * n + variable_input)

    cost = (
        input_tokens_billed / 1_000_000 * pricing.input_per_1m
        + output_tokens / 1_000_000 * pricing.output_per_1m
    )

    base_per_request = static_tokens + USER_WRAPPER_TOKENS + out_per
    avg_report = (sum(report_tokens) / n) if n else 0
    max_report = max(report_tokens) if report_tokens else 0

    return BatchEstimate(
        provider_key=provider_key,
        provider_label=provider_label,
        model=model,
        n_reports=n,
        static_tokens=static_tokens,
        input_tokens_billed=input_tokens_billed,
        output_tokens=output_tokens,
        cost_usd=round(cost, 4),
        free_tier=pricing.free_tier,
        supports_cache=pricing.supports_cache,
        seconds_low=n * pricing.seconds_per_report * 0.6,
        seconds_high=n * pricing.seconds_per_report * 1.6,
        tokens_per_request_avg=round(base_per_request + avg_report),
        tokens_per_request_max=round(base_per_request + max_report),
    )


def static_context_text(session) -> str:
    """The shared, cacheable context whose tokens are estimated once."""
    return build_grading_system_text(
        session.task_text,
        session.rubric_text,
        session.components,
        session.reference_samples,
    )


def compare_providers(
    session,
    documents: Sequence[ParsedDocument],
    presets: dict,
    pricing_table: dict | None = None,
) -> List[BatchEstimate]:
    """One estimate per provider preset (using its default model & pricing),
    sorted cheapest first — so the professor can pick the best-value option."""
    pricing_table = pricing_table or PROVIDER_PRICING
    ctx = static_context_text(session)
    report_texts = [d.text for d in documents]
    n_components = len(session.components)

    estimates: List[BatchEstimate] = []
    for key, preset in presets.items():
        if key == "custom":
            continue
        estimates.append(
            estimate_batch(
                provider_key=key,
                provider_label=preset.label,
                model=preset.default_model,
                static_context_text=ctx,
                report_texts=report_texts,
                n_components=n_components,
                pricing=pricing_table.get(key, DEFAULT_PRICING),
            )
        )
    estimates.sort(key=lambda e: (not e.free_tier, e.cost_usd))
    return estimates
