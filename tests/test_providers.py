"""Tests for provider-layer helpers that don't need network access."""
from __future__ import annotations

import pytest

from src.providers import PROVIDER_PRESETS, _extract_json, build_provider


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    text = '```json\n{"components": [{"name": "X", "weight": 100}]}\n```'
    data = _extract_json(text)
    assert data["components"][0]["name"] == "X"


def test_extract_json_with_prose_around():
    text = 'Here is the result:\n{"total": 65}\nHope that helps.'
    assert _extract_json(text) == {"total": 65}


def test_presets_have_required_fields():
    for key, p in PROVIDER_PRESETS.items():
        assert p.key == key
        assert p.kind in ("anthropic", "openai")


def test_build_provider_unknown():
    with pytest.raises(ValueError):
        build_provider("does-not-exist")


def test_build_provider_requires_model_for_custom():
    # Custom preset has no default model.
    with pytest.raises(ValueError):
        build_provider("custom", model=None)
