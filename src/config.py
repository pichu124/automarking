"""Central configuration: model selection and the statistical calibration targets.

The model string is read from the GRADING_MODEL environment variable so the
deployment can pick a model without code changes. The default is the most
capable current Claude model, which is what the grading persona is tuned for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --- Model / API ----------------------------------------------------------

DEFAULT_MODEL = os.getenv("GRADING_MODEL", "claude-opus-4-8")
DEFAULT_EFFORT = os.getenv("GRADING_EFFORT", "high")  # low | medium | high | max

# max_tokens for a single graded report. Grading output is structured JSON and
# bounded, so this is generous headroom rather than a streaming-scale ceiling.
GRADING_MAX_TOKENS = 8000
RUBRIC_MAX_TOKENS = 4000


# --- Context size caps (characters) ---------------------------------------
# Each grading request bundles the task, rubric, four reference samples and the
# student report. These caps are a SAFETY VALVE against pathologically large
# files (e.g. a 3.4MB PDF that extracts to hundreds of KB of text) and tight
# free-tier token-per-minute limits — NOT a routine trim. Defaults are generous:
# a normal full assignment (≈2,500 words ≈ 16k chars) and full reference papers
# are read in their entirety; only unusually long files are shortened, and the
# UI warns when that happens. ~4 chars/token. Set any cap to 0 to disable it
# (read everything) — best with large-context models (Claude, Gemini, Ollama);
# lower them only to fit small free tiers like Groq (~12k TPM).
MAX_TASK_CHARS = 16000         # ≈ 4k tokens
MAX_RUBRIC_CHARS = 16000       # ≈ 4k tokens
MAX_SAMPLE_CHARS = 24000       # per reference sample, ≈ 6k tokens (~3,700 words)
MAX_REPORT_CHARS = 40000       # per student report, ≈ 10k tokens (~6,000 words)


# --- Calibration targets (the "curve") ------------------------------------

@dataclass(frozen=True)
class CalibrationConfig:
    """The cohort distribution the Two-Pass engine enforces.

    Everything is expressed on a 0-100 percentage scale.
    """

    # Target cohort mean. The engine lands the final mean inside this band.
    target_mean_low: float = 65.0
    target_mean_high: float = 66.0

    # High Distinction: score >= hd_threshold, capped at hd_cap_fraction of N.
    hd_threshold: float = 80.0
    hd_cap_fraction: float = 0.10

    # Distinction: di_lower <= score < di_upper, capped at di_cap_fraction of N.
    di_lower: float = 70.0
    di_upper: float = 80.0
    di_cap_fraction: float = 0.20

    # Anything below this counts as a Fail; the engine minimises these.
    fail_threshold: float = 50.0

    # Per-rank score envelopes used to enforce the caps while preserving the
    # LLM's quality ranking. A soft floor at 50 is what "aggressively minimise
    # fails" translates to in practice.
    score_floor: float = 50.0
    score_ceiling: float = 95.0     # ceiling for the HD band
    di_ceiling: float = 79.0        # ranks in the DI band cannot reach 80
    base_ceiling: float = 69.0      # ranks below the DI band cannot reach 70


DEFAULT_CALIBRATION = CalibrationConfig()


# --- Pricing (for the pre-flight cost/time estimate) ----------------------

@dataclass(frozen=True)
class ModelPricing:
    """Approximate pricing + latency for a provider's default model.

    USD per 1,000,000 tokens. These move often and vary by exact model — they
    are INDICATIVE only and editable here. Free-tier providers are priced at
    their paid rate for the estimate but flagged ``free_tier`` so the UI can
    show "Free tier" instead of a dollar figure.
    """

    input_per_1m: float
    output_per_1m: float
    supports_cache: bool = False         # Claude prompt caching
    cache_read_mult: float = 0.1
    cache_write_mult: float = 1.25
    free_tier: bool = False
    seconds_per_report: float = 8.0      # rough per-report latency guess


# Keyed by provider key (see providers.PROVIDER_PRESETS). Approximate, ~2026.
PROVIDER_PRICING: dict[str, ModelPricing] = {
    "anthropic": ModelPricing(5.0, 25.0, supports_cache=True, seconds_per_report=22.0),
    "gemini": ModelPricing(0.10, 0.40, free_tier=True, seconds_per_report=5.0),
    "groq": ModelPricing(0.59, 0.79, free_tier=True, seconds_per_report=3.0),
    "ollama": ModelPricing(0.0, 0.0, free_tier=True, seconds_per_report=15.0),
    "openrouter": ModelPricing(0.60, 0.70, seconds_per_report=6.0),
    "deepseek": ModelPricing(0.28, 0.42, seconds_per_report=8.0),
    "mistral": ModelPricing(2.0, 6.0, seconds_per_report=6.0),
    "openai": ModelPricing(0.15, 0.60, seconds_per_report=6.0),
    "custom": ModelPricing(0.0, 0.0, seconds_per_report=8.0),
}

# Used when a provider key has no pricing entry.
DEFAULT_PRICING = ModelPricing(1.0, 4.0, seconds_per_report=8.0)
