"""Streamlit AppTest smoke: the real app must boot and survive every widget.

These run the actual app.py headlessly (no browser, no API calls) and catch
wiring bugs unit tests can't: bad session_state keys, widget defaults,
provider-preset branches, reset flow.
"""
from __future__ import annotations

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture()
def at():
    app = st_testing.AppTest.from_file("app.py", default_timeout=60)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    return app


def test_app_boots_with_all_providers_listed(at):
    assert len(at.sidebar.selectbox[0].options) >= 8


def test_every_provider_renders_without_exception(at):
    sb = at.sidebar.selectbox[0]
    for key in sb.options:
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
