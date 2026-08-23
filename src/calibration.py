"""Statistical calibration engine — the heart of the Two-Pass system.

Pass 1 (in ``grading_engine``) grades every report independently and produces a
raw total per report. This module is Pass 2: it maps those raw totals onto a
curve that simultaneously satisfies four hard constraints:

    * cohort mean lands inside [target_mean_low, target_mean_high]
    * #(score >= 80)            <= hd_cap_fraction * N      (HD cap)
    * #(70 <= score < 80)       <= di_cap_fraction * N      (DI cap)
    * #(score < 50)             aggressively minimised        (soft floor at 50)

Algorithm — "banded rank mapping with mean targeting":

  1. Rank reports by raw total (descending). The LLM's *ordering* is treated as
     ground truth; calibration only rescales magnitudes, never reorders.
  2. Assign each rank a score ceiling so the caps hold *by construction*:
       - top  floor(0.10*N) ranks  -> ceiling 95   (may be HD)
       - next floor(0.20*N) ranks  -> ceiling 79    (DI at most, never HD)
       - all remaining ranks       -> ceiling 69    (never DI/HD)
     Every rank shares a soft floor of 50 (minimise fails).
  3. Apply a single global additive shift `t` to every raw score, then clamp
     into each rank's [floor, ceiling]. Because raw scores and the bounds are
     all non-increasing in rank, the clamp preserves the ordering, and the mean
     is monotonic and continuous in `t` — so a binary search finds the `t` that
     hits the target mean exactly (when feasible).
  4. DI-overflow correction: a top-band report whose calibrated score lands in
     [70,80) counts toward the DI cap. If the DI band overflows, demote the
     lowest-ranked DI member's ceiling to 69 and re-solve. Each demotion
     permanently removes one rank from the DI-eligible set, so this terminates.

The construction guarantees the HD and DI caps and the no-reorder property; the
binary search guarantees the mean; the soft floor minimises fails.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Sequence

from .config import CalibrationConfig, DEFAULT_CALIBRATION


@dataclass
class CalibrationReport:
    """Diagnostics describing what the calibration did to the cohort."""

    n: int
    raw_mean: float
    final_mean: float
    hd_count: int
    di_count: int
    fail_count: int
    hd_cap: int
    di_cap: int
    shift: float
    raw_std: float = 0.0          # spread of the model's raw scores
    distinct_finals: int = 0      # how many distinct calibrated totals

    @property
    def low_discrimination(self) -> bool:
        """True when the model barely differentiated the cohort.

        A tiny raw spread means every report got ~the same score, so the
        calibrated grades cannot meaningfully rank students.
        """
        if self.n < 3:
            return False
        return self.raw_std < 6.0 or self.distinct_finals <= max(2, self.n // 4)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def calibrate_scores(
    raw_totals: Sequence[float],
    config: CalibrationConfig | None = None,
) -> tuple[List[float], CalibrationReport]:
    """Map raw Pass-1 totals to calibrated grades.

    Returns ``(final_totals, report)`` where ``final_totals`` is aligned with
    the input order.
    """
    cfg = config or DEFAULT_CALIBRATION
    n = len(raw_totals)
    if n == 0:
        return [], CalibrationReport(0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0)

    target_mean = (cfg.target_mean_low + cfg.target_mean_high) / 2.0

    # Rank positions by raw total (desc); the tie-break on -index keeps it stable.
    order = sorted(range(n), key=lambda i: (raw_totals[i], -i), reverse=True)
    sorted_raw = [float(raw_totals[i]) for i in order]

    n_hd = math.floor(cfg.hd_cap_fraction * n)
    n_di = math.floor(cfg.di_cap_fraction * n)

    ceil_by_rank: List[float] = []
    for pos in range(n):
        if pos < n_hd:
            ceil_by_rank.append(cfg.score_ceiling)
        elif pos < n_hd + n_di:
            ceil_by_rank.append(cfg.di_ceiling)
        else:
            ceil_by_rank.append(cfg.base_ceiling)

    floor = cfg.score_floor

    # Band cut-offs whose "just below" zone we keep scores out of. 50 is omitted
    # because the score_floor already prevents scores below it.
    boundary_thresholds = (cfg.credit_threshold, cfg.di_lower, cfg.hd_threshold)

    def avoid_boundaries(s: float) -> float:
        # Keep scores out of the contestable just-below-cut-off zone (59, 69,
        # 79): anything in [T - guard, T) is pushed down to T - guard, sitting
        # clearly inside the lower band. Monotonic (only ever pushes down, never
        # up across a threshold), so ranking, the HD/DI caps and the
        # mean-targeting binary search are all preserved.
        if not cfg.avoid_band_boundaries:
            return s
        g = cfg.boundary_guard
        for threshold in boundary_thresholds:
            if threshold - g <= s < threshold:
                return threshold - g
        return s

    def scores_for_shift(t: float) -> List[float]:
        # avoid_boundaries is monotonic (only pushes down, never up across a
        # threshold), so the ranking and the caps are preserved, and the cohort
        # mean stays monotonic in t — the binary search below still converges.
        return [
            avoid_boundaries(_clamp(sorted_raw[p] + t, floor, ceil_by_rank[p]))
            for p in range(n)
        ]

    def mean_for(t: float) -> float:
        s = scores_for_shift(t)
        return sum(s) / n

    def solve_shift() -> float:
        # mean_for is non-decreasing in t; binary-search the target.
        lo, hi = -1000.0, 1000.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if mean_for(mid) < target_mean:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    t = solve_shift()
    scores_sorted = scores_for_shift(t)

    # DI-overflow correction. Demote from the bottom of the DI band upward;
    # each demoted rank is capped below 70 permanently, so the loop terminates.
    for _ in range(n + 1):
        di_positions = [
            p for p in range(n) if cfg.di_lower <= scores_sorted[p] < cfg.di_upper
        ]
        if len(di_positions) <= n_di:
            break
        worst = max(di_positions)  # lowest-ranked report currently in the DI band
        ceil_by_rank[worst] = cfg.base_ceiling
        t = solve_shift()
        scores_sorted = scores_for_shift(t)

    scores_sorted = [round(s, 1) for s in scores_sorted]

    final = [0.0] * n
    for p, i in enumerate(order):
        final[i] = scores_sorted[p]

    hd_count = sum(1 for s in final if s >= cfg.hd_threshold)
    di_count = sum(1 for s in final if cfg.di_lower <= s < cfg.di_upper)
    fail_count = sum(1 for s in final if s < cfg.fail_threshold)

    report = CalibrationReport(
        n=n,
        raw_mean=round(sum(raw_totals) / n, 2),
        final_mean=round(sum(final) / n, 2),
        hd_count=hd_count,
        di_count=di_count,
        fail_count=fail_count,
        hd_cap=n_hd,
        di_cap=n_di,
        shift=round(t, 2),
        raw_std=round(statistics.pstdev(raw_totals), 2) if n > 1 else 0.0,
        distinct_finals=len(set(final)),
    )
    return final, report


def rescale_components(
    scores: Sequence[float],
    max_scores: Sequence[float],
    old_total: float,
    new_total: float,
) -> List[float]:
    """Rescale component scores so they sum to ``new_total`` after calibration.

    Each component is scaled by ``new_total / old_total`` and clamped to
    ``[0, max_score]``; any rounding drift is absorbed by the component with
    the most headroom so the sum matches ``new_total`` exactly. Inputs are
    clamped first, so impossible model output (negative scores, or scores above
    the component maximum) cannot propagate into the exported grades.
    """
    maxes = [float(m) for m in max_scores]
    scores = [_clamp(float(s), 0.0, maxes[i]) for i, s in enumerate(scores)]
    if old_total <= 0:
        total_max = sum(maxes) or 1.0
        out = [round(new_total * m / total_max, 1) for m in maxes]
    else:
        factor = new_total / old_total
        out = [round(_clamp(scores[i] * factor, 0.0, maxes[i]), 1) for i in range(len(scores))]

    # Absorb drift into the component with the largest remaining headroom.
    drift = round(new_total - sum(out), 1)
    if abs(drift) >= 0.1 and out:
        if drift > 0:
            idx = max(range(len(out)), key=lambda i: maxes[i] - out[i])
        else:
            idx = max(range(len(out)), key=lambda i: out[i])
        out[idx] = round(_clamp(out[idx] + drift, 0.0, maxes[idx]), 1)
    return out
