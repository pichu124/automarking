"""Multi-provider grading back-ends.

Anthropic's API is paid, so the app supports several alternatives — including
free tiers (Google Gemini, Groq) and a fully-free local option (Ollama). Most
providers speak the OpenAI Chat Completions protocol, so a single
``OpenAICompatibleProvider`` (the ``openai`` SDK pointed at a different
``base_url``) covers Gemini, Groq, OpenRouter, Ollama, DeepSeek, Mistral,
Together and plain OpenAI. The native ``AnthropicProvider`` keeps Claude's
prompt caching.

Every provider exposes the same two methods, so ``grading_engine`` treats them
interchangeably:

    structure_rubric(rubric_text) -> StructuredRubric
    grade_report(...)             -> GradingResult
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from .config import GRADING_MAX_TOKENS, RUBRIC_MAX_TOKENS
from .llm_client import GradingLLM, _normalise_weights
from .models import GradingResult, RubricComponent, StructuredRubric
from .prompts import (
    GRADING_JSON_INSTRUCTION,
    RUBRIC_JSON_INSTRUCTION,
    RUBRIC_STRUCTURING_PROMPT,
    build_grading_system_text,
    build_grading_user_message,
)


# --- Provider catalogue ----------------------------------------------------

@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    kind: str                 # "anthropic" | "openai"
    default_model: str
    api_key_env: Optional[str]
    base_url: Optional[str] = None
    free: bool = False
    note: str = ""
    api_key_optional: bool = False  # e.g. local Ollama needs no real key


# Ordered so the UI can show free/local options prominently.
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "anthropic": ProviderPreset(
        key="anthropic", label="Anthropic Claude", kind="anthropic",
        default_model="claude-opus-4-8", api_key_env="ANTHROPIC_API_KEY",
        note="Highest grading quality; prompt caching. Paid.",
    ),
    "gemini": ProviderPreset(
        key="gemini", label="Google Gemini (free tier)", kind="openai",
        default_model="gemini-2.0-flash", api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        free=True, note="Generous free tier. Get a key at aistudio.google.com.",
    ),
    "groq": ProviderPreset(
        key="groq", label="Groq (free tier)", kind="openai",
        default_model="llama-3.3-70b-versatile", api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        free=True, note="Fast, free tier; open models. console.groq.com.",
    ),
    "ollama": ProviderPreset(
        key="ollama", label="Ollama (local, free)", kind="openai",
        default_model="llama3.1", api_key_env=None,
        base_url="http://localhost:11434/v1",
        free=True, api_key_optional=True,
        note="Runs fully offline on your machine. Install from ollama.com.",
    ),
    "openrouter": ProviderPreset(
        key="openrouter", label="OpenRouter", kind="openai",
        default_model="meta-llama/llama-3.3-70b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        note="Aggregator; some free models (try ':free' model ids).",
    ),
    "deepseek": ProviderPreset(
        key="deepseek", label="DeepSeek", kind="openai",
        default_model="deepseek-chat", api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        note="Very low cost.",
    ),
    "mistral": ProviderPreset(
        key="mistral", label="Mistral", kind="openai",
        default_model="mistral-large-latest", api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        note="Free/dev tier available.",
    ),
    "openai": ProviderPreset(
        key="openai", label="OpenAI", kind="openai",
        default_model="gpt-4o-mini", api_key_env="OPENAI_API_KEY",
        note="gpt-4o-mini is cheap and capable.",
    ),
    "custom": ProviderPreset(
        key="custom", label="Custom (OpenAI-compatible)", kind="openai",
        default_model="", api_key_env=None, base_url="",
        api_key_optional=True, note="Point at any OpenAI-compatible endpoint.",
    ),
}


# --- JSON helpers ----------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating fences/prose."""
    if text is None:
        raise ValueError("empty model response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # drop the opening fence (``` or ```json) and the closing fence
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


# --- Anthropic (native, cached) -------------------------------------------

class AnthropicProvider:
    """Native Claude path — keeps prompt caching + adaptive thinking."""

    def __init__(self, model: str, api_key: str | None, effort: str):
        self.llm = GradingLLM(api_key=api_key or None, model=model, effort=effort)

    def structure_rubric(self, rubric_text: str) -> StructuredRubric:
        return self.llm.structure_rubric(rubric_text)

    def grade_report(self, **kwargs) -> GradingResult:
        return self.llm.grade_report(**kwargs)


# --- OpenAI-compatible (Gemini / Groq / Ollama / OpenRouter / ...) ---------

class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float = 0.2,
        max_tokens_grade: int = GRADING_MAX_TOKENS,
        max_tokens_rubric: int = RUBRIC_MAX_TOKENS,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'openai' package is required for non-Anthropic providers. "
                "Install it with: pip install openai"
            ) from exc
        self.model = model
        self.temperature = temperature
        self.max_tokens_grade = max_tokens_grade
        self.max_tokens_rubric = max_tokens_rubric
        # OpenAI client requires a non-empty key even for keyless local servers.
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)

    def _chat_json(self, system: str, user: str, max_tokens: int) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        # Prefer JSON mode; fall back if the provider/model doesn't support it.
        try:
            resp = self.client.chat.completions.create(
                response_format={"type": "json_object"}, **base_kwargs
            )
        except Exception:
            resp = self.client.chat.completions.create(**base_kwargs)
        return _extract_json(resp.choices[0].message.content)

    def structure_rubric(self, rubric_text: str) -> StructuredRubric:
        system = f"{RUBRIC_STRUCTURING_PROMPT}\n\n# OUTPUT FORMAT\n{RUBRIC_JSON_INSTRUCTION}"
        user = f"Structure this rubric into weighted components:\n\n{rubric_text.strip()}"
        data = self._chat_json(system, user, self.max_tokens_rubric)
        rubric = StructuredRubric.model_validate(data)
        return _normalise_weights(rubric)

    def grade_report(
        self,
        *,
        task_text: str,
        rubric_text: str,
        components: List[RubricComponent],
        reference_samples: dict[str, str],
        source_name: str,
        report_text: str,
    ) -> GradingResult:
        system = build_grading_system_text(
            task_text, rubric_text, components, reference_samples
        )
        user = build_grading_user_message(source_name, report_text)
        data = self._chat_json(system, user, self.max_tokens_grade)
        return GradingResult.model_validate(data)


# --- Factory ---------------------------------------------------------------

def build_provider(
    provider_key: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    effort: str = "high",
    temperature: float = 0.2,
):
    """Construct the grader for a provider key (see PROVIDER_PRESETS)."""
    preset = PROVIDER_PRESETS.get(provider_key)
    if preset is None:
        raise ValueError(f"Unknown provider: {provider_key}")

    model = model or preset.default_model
    if not model:
        raise ValueError("A model name is required.")

    if api_key is None and preset.api_key_env:
        api_key = os.getenv(preset.api_key_env)
    resolved_base_url = base_url if base_url is not None else preset.base_url

    if preset.kind == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, effort=effort)
    return OpenAICompatibleProvider(
        model=model,
        api_key=api_key,
        base_url=resolved_base_url,
        temperature=temperature,
    )
