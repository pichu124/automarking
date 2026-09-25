"""Tests for the calibration engine — the constraints that must always hold.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import math
import random

from src.config import DEFAULT_CALIBRATION as CFG
from src.calibration import calibrate_scores, rescale_components


def _assert_constraints(raw, final, report):
    n = len(raw)
    # Mean lands inside the target band.
    assert CFG.target_mean_low - 0.2 <= report.final_mean <= CFG.target_mean_high + 0.2, (
        f"mean {report.final_mean} outside band"
    )
    # HD cap.
    hd = sum(1 for s in final if s >= CFG.hd_threshold)
    assert hd <= math.floor(CFG.hd_cap_fraction * n), f"HD {hd} exceeds cap"
    # DI cap.
    di = sum(1 for s in final if CFG.di_lower <= s < CFG.di_upper)
    # With no HD in the cohort the DI cap widens (di_cap_fraction_no_hd).
    di_frac = CFG.di_cap_fraction_no_hd if report.di_cap_widened else CFG.di_cap_fraction
    assert report.di_cap == math.floor(di_frac * n)
    assert di <= report.di_cap, f"DI {di} exceeds cap"
    # Scores stay in range.
    assert all(CFG.score_floor - 0.01 <= s <= CFG.score_ceiling + 0.01 for s in final)
    # Ranking is preserved: sorting raw desc gives final non-increasing.
    order = sorted(range(n), key=lambda i: raw[i], reverse=True)
    ordered_final = [final[i] for i in order]
    assert all(
        ordered_final[k] >= ordered_final[k + 1] - 0.11 for k in range(n - 1)
    ), "ranking not preserved"


def test_normal_spread():
    random.seed(1)
    raw = [random.gauss(70, 12) for _ in range(40)]
    raw = [max(0, min(100, x)) for x in raw]
    final, report = calibrate_scores(raw)
    _assert_constraints(raw, final, report)


def test_all_high_compresses_down():
    # LLM was generous: everyone ~88. Must compress to the target mean.
    raw = [88.0 + i * 0.1 for i in range(30)]
    final, report = calibrate_scores(raw)
    _assert_constraints(raw, final, report)
    assert report.final_mean <= CFG.target_mean_high + 0.2


def test_all_low_lifts_up_and_minimises_fails():
    # LLM was harsh: everyone failing. Must lift mean and minimise fails.
    raw = [25.0 + i * 0.5 for i in range(30)]
    final, report = calibrate_scores(raw)
    _assert_constraints(raw, final, report)
    assert report.fail_count == 0, "soft floor should eliminate fails here"


def test_compressed_top_band_di_overflow():
    # Many reports clustered in the 70s — would overflow the DI band without
    # the demotion correction.
    raw = [78.0 - i * 0.05 for i in range(50)]
    final, report = calibrate_scores(raw)
    _assert_constraints(raw, final, report)


def test_small_cohort():
    for n in (1, 2, 3, 5):
        raw = [60.0 + i for i in range(n)]
        final, report = calibrate_scores(raw)
        _assert_constraints(raw, final, report)


def test_identical_scores():
    raw = [70.0] * 25
    final, report = calibrate_scores(raw)
    _assert_constraints(raw, final, report)
    # All equal -> all land on the target mean midpoint.
    assert all(abs(s - 65.5) < 0.2 for s in final)


def test_boundary_values_are_avoided():
    # The frustrating "just below a band cut-off" scores (59, 69, 79) must not
    # appear; nothing may sit in (T - guard, T) for T in {60, 70, 80}.
    guard = CFG.boundary_guard
    for seed in range(20):
        random.seed(seed)
        n = random.randint(5, 40)
        raw = [max(0, min(100, random.gauss(75, 12))) for _ in range(n)]
        final, _ = calibrate_scores(raw)
        for f in final:
            for threshold in (60.0, 70.0, 80.0):
                assert not (threshold - guard + 0.01 < f < threshold), (
                    f"score {f} sits in the guard zone below {threshold}"
                )
        _assert_constraints(raw, final, calibrate_scores(raw)[1])


def test_boundary_cluster_pushed_below_seventy():
    # The real case: many reports capped just under Distinction end up at the
    # top of Credit — they must land at 68.5, never 69.
    raw = [88.0 + (i % 4) * 0.2 for i in range(16)]
    final, _ = calibrate_scores(raw)
    assert all(f <= 68.5 or f >= 70.0 for f in final)
    assert not any(69.0 <= f < 70.0 for f in final)


def test_empty():
    final, report = calibrate_scores([])
    assert final == []
    assert report.n == 0


def test_low_discrimination_flagged_when_model_uniform():
    # Model gave everyone ~88 (the real-world failure mode) -> tiny spread.
    raw = [88.0 + (i % 2) * 0.3 for i in range(12)]
    _, report = calibrate_scores(raw)
    assert report.raw_std < 6.0
    assert report.low_discrimination is True


def test_low_discrimination_not_flagged_for_real_spread():
    random.seed(7)
    raw = [max(0, min(100, random.gauss(70, 14))) for _ in range(30)]
    _, report = calibrate_scores(raw)
    assert report.low_discrimination is False


def test_rescale_components_sums_to_total():
    scores = [30.0, 25.0, 20.0]
    maxes = [40.0, 30.0, 30.0]
    new = rescale_components(scores, maxes, old_total=75.0, new_total=66.0)
    assert abs(sum(new) - 66.0) < 0.11
    assert all(0 <= new[i] <= maxes[i] for i in range(3))


def test_rescale_components_zero_old_total():
    scores = [0.0, 0.0]
    maxes = [50.0, 50.0]
    new = rescale_components(scores, maxes, old_total=0.0, new_total=60.0)
    assert abs(sum(new) - 60.0) < 0.11


def test_fuzz_invariants_across_sidebar_configs():
    """Seeded sweep over every sidebar-reachable config x cohort shape.

    Pins the two defects fuzzing found: mid-sequence ceiling demotion breaking
    rank monotonicity (higher raw -> lower final), and band counts checked on
    unrounded scores (69.96 rounding up into the DI band after the check).
    """
    import dataclasses
    import itertools

    random.seed(7)
    cohorts = []
    for n in (1, 3, 8, 21, 40):
        cohorts.append([min(100, max(0, random.gauss(70, 12))) for _ in range(n)])
        cohorts.append([min(100, max(0, random.gauss(88, 3))) for _ in range(n)])
        cohorts.append([min(100, max(0, random.gauss(30, 8))) for _ in range(n)])
        cohorts.append([70.0] * n)
    configs = itertools.product(
        (55.0, 65.5, 75.0), (0, 10, 30), (0, 20, 40),
        ((True, 1.5), (True, 3.0), (False, 1.5)),
    )
    for mean, hd, di, (avoid, guard) in configs:
        cfg = dataclasses.replace(
            CFG, target_mean_low=mean - 0.5, target_mean_high=mean + 0.5,
            hd_cap_fraction=hd / 100, di_cap_fraction=di / 100,
            avoid_band_boundaries=avoid, boundary_guard=guard,
        )
        for raw in cohorts:
            n = len(raw)
            final, rep = calibrate_scores(raw, cfg)
            assert sum(1 for s in final if s >= cfg.hd_threshold) <= math.floor(hd / 100 * n)
            di_frac = max(di / 100, cfg.di_cap_fraction_no_hd) if rep.di_cap_widened else di / 100
            if rep.di_cap_widened:
                assert rep.hd_count == 0
            assert sum(1 for s in final if cfg.di_lower <= s < cfg.di_upper) <= math.floor(di_frac * n)
            assert all(cfg.score_floor - 0.01 <= s <= cfg.score_ceiling + 0.01 for s in final)
            if avoid:
                assert not any(
                    T - guard + 0.05 < s < T - 0.05
                    for s in final for T in (60.0, 70.0, 80.0)
                )
            order = sorted(range(n), key=lambda i: raw[i], reverse=True)
            of = [final[i] for i in order]
            assert all(of[k] >= of[k + 1] - 0.11 for k in range(n - 1)), (
                f"rank inversion: cfg=({mean},{hd},{di},{avoid},{guard})"
            )


def test_no_hd_places_widens_di_cap_to_30_percent():
    # 7 reports: the HD cap (10%) rounds to 0 places, so DI may take 30%
    # (2 places) instead of 20% (1 place).
    raw = [82.0, 81.5, 81.0, 76.0, 74.0, 70.0, 62.0]
    final, report = calibrate_scores(raw)
    assert report.hd_count == 0
    assert report.di_cap_widened is True
    assert report.di_cap == math.floor(0.30 * len(raw)) == 2
    assert sum(1 for s in final if 70 <= s < 80) <= 2


def test_di_cap_stays_20_percent_when_cohort_has_hd():
    random.seed(3)
    raw = [max(0, min(100, random.gauss(72, 12))) for _ in range(30)]
    final, report = calibrate_scores(raw)
    assert report.hd_count > 0
    assert report.di_cap_widened is False
    assert report.di_cap == math.floor(0.20 * 30)


def test_no_hd_awarded_despite_places_also_widens():
    import dataclasses

    # HD places exist, but the target mean is so low nobody reaches 80.
    cfg = dataclasses.replace(CFG, target_mean_low=55.0, target_mean_high=56.0)
    raw = [60.0 + i * 0.5 for i in range(20)]
    final, report = calibrate_scores(raw, cfg)
    if report.hd_count == 0:
        assert report.di_cap_widened is True
        assert report.di_cap == math.floor(0.30 * 20)
