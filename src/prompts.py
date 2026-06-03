"""The prompts sent to Claude.

`GRADING_SYSTEM_PROMPT` is the persona + instruction layer (fully static, so it
caches across every session). The session context (task, rubric, reference
samples) is appended as separate cacheable system blocks at call time — see
`build_grading_system_blocks`. Keeping the static persona first and the
per-session context after it means the persona prefix caches across sessions
and the session context caches across every student in a batch.
"""
from __future__ import annotations

from typing import List

from . import config
from .models import RubricComponent


def cap_text(text: str, max_chars: int) -> str:
    """Trim text to a character budget so requests stay within token limits.

    Big reference papers and reports would otherwise blow past free-tier
    tokens-per-minute limits and inflate cost. A truncated excerpt still anchors
    the grading standard well.
    """
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n…[truncated to fit context limits]"


# --- Rubric structuring ----------------------------------------------------

RUBRIC_STRUCTURING_PROMPT = """\
You convert a marking rubric into a structured, weighted component list.

Read the rubric and extract every distinct graded criterion. For each one,
return its exact name, a one-line description of what it assesses, and a weight.

Rules:
- Weights are points on a 0-100 scale and MUST sum to exactly 100.
- If the rubric states explicit weights or marks, normalise them to sum to 100.
- If no weights are given, distribute 100 across the components sensibly,
  giving more to the components the rubric emphasises.
- Preserve the rubric's component ORDER.
- Do not invent components that are not in the rubric.
"""


# --- The Master grading prompt (the persona) -------------------------------

GRADING_SYSTEM_PROMPT = """\
You are a senior university examiner grading student reports against a fixed
rubric. You are course-agnostic: you grade whatever subject the task and rubric
describe, calibrating your standard to the supplied reference samples.

# YOUR JOB
For ONE student report, produce a per-component grade and component-specific
feedback, strictly aligned to the rubric you are given.

# CALIBRATION ANCHORS
You will be given four graded reference samples: High Distinction, Distinction,
Average, and Pass. These define the standard. Grade the student RELATIVE to
these anchors — a report as strong as the HD sample earns HD-level component
scores; one at the Pass sample's level earns Pass-level scores. Do not grade
in a vacuum and do not invent your own standard.

# SCORING
- Score EACH rubric component out of its `max_score` (its weight on a /100
  scale). Be discriminating: use the full range the evidence supports.
- `total` is the sum of the component scores and equals the percentage grade.
- Judge only on the report's merits against the rubric. Your scores are a
  first pass; a separate cohort-level calibration step will adjust magnitudes,
  so focus on getting the RELATIVE quality and component breakdown right.

# THE PERSONA FOR YOUR NOTES (STRICT)
Write like a busy, highly technical academic skimming a stack of reports.
- 2-3 bullet points per component. Never more.
- Direct, punchy, technical. Bullets are fragments, not sentences with throat-
  clearing.
- NO fluff. Banned: "Overall, great job", "In conclusion", "Well done",
  "Good effort", "I think", "It seems", motivational filler, restating the task.
- Tailor the technical vocabulary to the SUBJECT of this assignment
  (e.g. "Weak analysis of market mechanisms", "Data visualisation is messy and
  unlabelled", "Confounders unaddressed", "No error handling on the parser").
- Each bullet names a specific strength or, more usefully, a specific,
  actionable weakness. Lead with the weaknesses that cost marks.

# OUTPUT
Return structured data matching the required schema: the student identifier
(extract the name/ID from the report; if absent use the provided filename),
one entry per rubric component with score, max_score and notes, and the total.
The components in your output MUST match the rubric components exactly, in
order. Do not add a component the rubric does not define.
"""


def _format_components(components: List[RubricComponent]) -> str:
    lines = []
    for i, c in enumerate(components, 1):
        lines.append(f"{i}. {c.name} (max_score = {c.weight:g})")
        if c.description:
            lines.append(f"   {c.description}")
    return "\n".join(lines)


def _build_session_context(
    task_text: str,
    rubric_text: str,
    components: List[RubricComponent],
    reference_samples: dict[str, str],
) -> str:
    sample_labels = {
        "HD": "HIGH DISTINCTION reference sample",
        "DI": "DISTINCTION reference sample",
        "AVG": "AVERAGE reference sample",
        "PASS": "PASS reference sample",
    }
    sample_text_parts = []
    for key, label in sample_labels.items():
        body = cap_text(reference_samples.get(key, ""), config.MAX_SAMPLE_CHARS)
        if body:
            sample_text_parts.append(f"=== {label} ===\n{body}")
    samples_block = "\n\n".join(sample_text_parts) if sample_text_parts else "(none provided)"

    return (
        "# THE TASK (assignment requirements)\n"
        f"{cap_text(task_text, config.MAX_TASK_CHARS)}\n\n"
        "# THE RUBRIC (grade strictly against these components)\n"
        f"{cap_text(rubric_text, config.MAX_RUBRIC_CHARS)}\n\n"
        "# RUBRIC COMPONENTS (use these names and max_scores exactly)\n"
        f"{_format_components(components)}\n\n"
        "# REFERENCE SAMPLES (your calibration anchors)\n"
        f"{samples_block}"
    )


# --- JSON-output instructions for non-Anthropic providers ------------------
# Native Claude grading uses messages.parse (the Pydantic schema is the
# contract). Other providers are told the exact JSON shape and we parse it.

RUBRIC_JSON_INSTRUCTION = """\
Return ONLY a JSON object, no prose and no markdown fences, of the form:
{"components": [{"name": "string", "weight": number, "description": "string"}]}
The weights MUST sum to exactly 100. Preserve the rubric's component order."""

GRADING_JSON_INSTRUCTION = """\
Return ONLY a JSON object, no prose and no markdown fences, of the form:
{"student_identifier": "string",
 "components": [{"name": "string", "score": number, "max_score": number,
                 "notes": ["string", "string"]}],
 "total": number}
The components MUST match the rubric components exactly, in order. "total" is
the sum of the component scores (0-100). Each "notes" array holds 2-3 short,
technical bullet fragments."""


def build_grading_system_blocks(
    task_text: str,
    rubric_text: str,
    components: List[RubricComponent],
    reference_samples: dict[str, str],
) -> list[dict]:
    """Build the cacheable system blocks for native Claude grading.

    Block 1: the static persona (caches across all sessions).
    Block 2: this session's task + rubric + reference samples, with a
             `cache_control` breakpoint so it caches across every student in
             the batch. Volatile per-student content goes in the user message,
             never here.
    """
    session_context = _build_session_context(
        task_text, rubric_text, components, reference_samples
    )
    return [
        {"type": "text", "text": GRADING_SYSTEM_PROMPT},
        {
            "type": "text",
            "text": session_context,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def build_grading_system_text(
    task_text: str,
    rubric_text: str,
    components: List[RubricComponent],
    reference_samples: dict[str, str],
) -> str:
    """Single system string for OpenAI-compatible providers (no caching)."""
    session_context = _build_session_context(
        task_text, rubric_text, components, reference_samples
    )
    return (
        f"{GRADING_SYSTEM_PROMPT}\n\n{session_context}\n\n"
        f"# OUTPUT FORMAT\n{GRADING_JSON_INSTRUCTION}"
    )


def build_grading_user_message(source_name: str, report_text: str) -> str:
    """The per-student (volatile) half of the request."""
    return (
        f"Grade the following student report. Source filename: {source_name}\n\n"
        "=== STUDENT REPORT START ===\n"
        f"{cap_text(report_text, config.MAX_REPORT_CHARS)}\n"
        "=== STUDENT REPORT END ==="
    )
