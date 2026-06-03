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
