"""Streamlit AppTest smoke: the real app must boot and survive every widget.

These run the actual app.py headlessly (no browser, no API calls) and catch
wiring bugs unit tests can't: bad session_state keys, widget defaults,
provider-preset branches, reset flow.
"""
from __future__ import annotations

import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture()
def at():
    app = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    return app


def test_app_boots_with_all_providers_listed(at):
    assert len(at.sidebar.selectbox[0].options) >= 8


def test_every_provider_renders_without_exception(at):
    from src.providers import PROVIDER_PRESETS

    # AppTest's `options` are the format_func labels (Streamlit >= 1.4x), so
    # drive the widget by the raw preset keys instead.
    sb = at.sidebar.selectbox[0]
    assert len(sb.options) == len(PROVIDER_PRESETS)
    for key in PROVIDER_PRESETS:
        sb.set_value(key)
        at.run()
        assert not at.exception, f"provider {key}: {at.exception[0] if at.exception else ''}"


def test_calibration_slider_extremes(at):
    for label, value in [
        ("Target mean", 55.0), ("Target mean", 75.0),
        ("HD cap", 0), ("HD cap", 30),
        ("Distinction cap", 0), ("Distinction cap", 40),
        ("Boundary buffer", 3.0),
    ]:
        sliders = [s for s in at.sidebar.slider if label.lower() in s.label.lower()]
        assert sliders, f"slider '{label}' missing from sidebar"
        sliders[0].set_value(value)
        at.run()
        assert not at.exception, f"{label}={value}"


def test_boundary_toggle_and_zero_caps(at):
    checkbox = [c for c in at.sidebar.checkbox if "boundary" in c.label.lower()][0]
    checkbox.set_value(False)
    at.run()
    checkbox.set_value(True)
    at.run()
    for ni in at.sidebar.number_input:
        ni.set_value(0)
        at.run()
    assert not at.exception


def test_new_session_reset(at):
    button = [b for b in at.sidebar.button if "New Grading Session" in b.label][0]
    button.click()
    at.run()
    assert not at.exception


def test_consecutive_edits_are_never_reverted(at):
    """Keyed-widget regression: without stable keys, Streamlit derives widget
    IDs from the value, so every SECOND consecutive edit was silently dropped
    (the professor's typo fix vanished; a cap set back to 0 stayed capped)."""
    def task_area():
        return [w for w in at.text_area if "Task text" in w.label][0]

    task_area().set_value("Task v1 (with a tpyo)")
    at.run()
    assert at.session_state["task_text"] == "Task v1 (with a tpyo)"
    task_area().set_value("Task v2 (typo fixed)")
    at.run()
    assert at.session_state["task_text"] == "Task v2 (typo fixed)"
    task_area().set_value("Task v3")
    at.run()
    assert at.session_state["task_text"] == "Task v3"

    def cap_input():
        return [w for w in at.sidebar.number_input if "Student report" in w.label][0]

    cap_input().set_value(2000)
    at.run()
    assert at.session_state["report_cap"] == 2000
    cap_input().set_value(0)  # back to "read everything"
    at.run()
    assert at.session_state["report_cap"] == 0

    def model_input():
        return [w for w in at.sidebar.text_input if w.label == "Model"][0]

    model_input().set_value("model-a")
    at.run()
    model_input().set_value("model-b")
    at.run()
    assert at.session_state["model"] == "model-b"


def test_cap_override_does_not_leak_into_next_session_defaults(at):
    """One session's cap override must not become another session's default."""
    import streamlit.testing.v1 as st_testing_v1

    from src import config

    cap = [w for w in at.sidebar.number_input if "Student report" in w.label][0]
    cap.set_value(2000)
    at.run()
    assert config.MAX_REPORT_CHARS == 2000

    fresh = st_testing_v1.AppTest.from_file(str(APP), default_timeout=60)
    fresh.run()
    fresh_cap = [w for w in fresh.sidebar.number_input if "Student report" in w.label][0]
    assert fresh_cap.value == config.BASE_MAX_REPORT_CHARS
