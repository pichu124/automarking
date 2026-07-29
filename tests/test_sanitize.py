"""Tests for keyboard-only note sanitisation (no smart quotes/dashes/etc.)."""
from __future__ import annotations

from src.docx_export import sanitize_text


def test_user_example_characters():
    before = (
        "Firm data present (≈$600bn assets, $1.8bn Q1 net income, NII +6%, "
        "NIM +11bp, FirstBank acquisition) but description crammed into a single "
        "paragraph with no distinct industry characterisation"
    )
    after = sanitize_text(before)
    assert "≈" not in after and "~$600bn" in after


def test_en_dash_becomes_hyphen():
    s = sanitize_text("No theoretical framing at this stage – concept surfaces later")
    assert "–" not in s
    assert " - concept" in s


def test_smart_quotes_arrows_ellipsis():
    s = sanitize_text("“clear” ‘risk’ hedge → payoff…")
    assert s == '"clear" \'risk\' hedge -> payoff...'


def test_result_is_pure_ascii():
    messy = "–—‘’“”…≈≠≤≥±×→ €600 °C Phạm"
    out = sanitize_text(messy)
    assert out.isascii(), f"non-ASCII left: {out!r}"


def test_transliterates_accents():
    assert "Pham" in sanitize_text("Phạm's report")  # ạ -> a


def test_empty_and_plain_unchanged():
    assert sanitize_text("") == ""
    assert sanitize_text("Weak analysis of market mechanisms") == (
        "Weak analysis of market mechanisms"
    )
