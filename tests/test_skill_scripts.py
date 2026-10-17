"""Calibration caps: floor(fraction * N) must not lose a place to float error."""
from __future__ import annotations

import dataclasses

import pytest

from src.calibration import calibrate_scores
from src.config import DEFAULT_CALIBRATION


@pytest.mark.parametrize("n,field,frac,expected", [
    (100, "hd_cap_fraction", 0.29, 29),  # 0.29 * 100 == 28.999999999999996
    (180, "di_cap_fraction", 0.35, 63),  # 0.35 * 180 == 62.99999999999999
])
def test_caps_survive_float_rounding(n, field, frac, expected):
    cfg = dataclasses.replace(DEFAULT_CALIBRATION, **{field: frac})
    _, rep = calibrate_scores([float(i % 100) for i in range(n)], cfg)
    assert (rep.hd_cap if field.startswith("hd") else rep.di_cap) == expected
