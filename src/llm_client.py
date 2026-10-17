"""Thin wrapper around the Anthropic SDK for the two grading calls.

Both calls use:
  * ``claude-opus-4-8`` (configurable) — the most capable model, which the
    grading persona is tuned for.
  * adaptive thinking + a configurable effort level.
  * structured outputs via ``messages.parse`` (Pydantic schema), so the
    response is a validated object, not free text we have to parse.

The grading call additionally relies on **prompt caching**: the static persona
and the per-session task/rubric/reference-sample context live in the ``system``
blocks with a ``cache_control`` breakpoint, so every student in a batch reads
that large prefix from cache (~0.1x cost) instead of re-sending it.
"""
from __future__ import annotations

import anthropic

from . import config
from .models import GradingResult, StructuredRubric
from .prompts import (
    RUBRIC_STRUCTURING_PROMPT,
    build_grading_system_blocks,
    build_grading_user_message,
    cap_text,
)


class ModelOutputError(ValueError):
    """The model produced unusable output for ONE report.

    A distinct type so the fallback engine can classify it as a per-report
    content error no matter what the message contains — the message embeds the
    model's own prose, which could coincidentally include health-sounding words
    ("billing", "quota") when the student's report is about such a topic.
    """


class GradingLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_retries: int = 4,
    ):
        # The SDK reads ANTHROPIC_API_KEY from the environment when api_key is
        # None. max_retries lets the SDK ride out transient 429/5xx with its own
        # exponential backoff before we consider falling back to another provider.
        kwargs = {"max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model or config.DEFAULT_MODEL
        self.effort = effort or config.DEFAULT_EFFORT

    # --- Pass 0: structure the rubric -------------------------------------

    def structure_rubric(self, rubric_text: str) -> StructuredRubric:
        """Parse the uploaded rubric into ordered, weighted components."""
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=config.RUBRIC_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=RUBRIC_STRUCTURING_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Structure this rubric into weighted components:\n\n"
                        f"{cap_text(rubric_text, config.MAX_RUBRIC_CHARS)}"
                    ),
                }
            ],
            output_format=StructuredRubric,
        )
        rubric = response.parsed_output
        if rubric is None:
            raise ModelOutputError("Rubric structuring returned no parseable output.")
        return _normalise_weights(rubric)

    # --- Generic text completion (used for sample summarisation) ----------

    def complete_text(self, system: str, user: str, max_tokens: int = 1200) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    # --- Pass 1: grade one report -----------------------------------------

    def grade_report(
        self,
        *,
        task_text: str,
        rubric_text: str,
        components,
        reference_samples: dict[str, str],
        source_name: str,
        report_text: str,
        marker_materials: str = "",
    ) -> GradingResult:
        """Grade a single report against the cached session context."""
        system_blocks = build_grading_system_blocks(
            task_text, rubric_text, components, reference_samples, marker_materials
        )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=config.GRADING_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=system_blocks,
            messages=[
                {
                    "role": "user",
                    "content": build_grading_user_message(source_name, report_text),
                }
            ],
            output_format=GradingResult,
        )
        result = response.parsed_output
        if result is None:
            raise ModelOutputError(f"Grading returned no parseable output for {source_name}.")
        return result


def _normalise_weights(rubric: StructuredRubric) -> StructuredRubric:
    """Force component weights to sum to exactly 100 (guards against LLM drift)."""
    total = sum(c.weight for c in rubric.components)
    if total <= 0:
        even = 100.0 / max(1, len(rubric.components))
        for c in rubric.components:
            c.weight = round(even, 2)
        return rubric
    if abs(total - 100.0) > 0.01:
        for c in rubric.components:
            c.weight = round(c.weight * 100.0 / total, 2)
    return rubric
