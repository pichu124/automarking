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
    di_cap_widened: bool = False  # no-HD rule applied (DI cap raised)

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

    If the cohort ends up with no High Distinction - the HD cap rounds down to
    zero places, or no report reaches the HD threshold - the unused HD share
    is released to Distinction: the cohort is re-calibrated with the DI cap at
    ``di_cap_fraction_no_hd`` (and the report's ``di_cap_widened`` is set).
    """
    cfg = config or DEFAULT_CALIBRATION
    final, report = _calibrate_once(raw_totals, cfg)
    wider = getattr(cfg, "di_cap_fraction_no_hd", cfg.di_cap_fraction)
    if report.n and report.hd_count == 0 and wider > cfg.di_cap_fraction:
        import dataclasses

        w_final, w_report = _calibrate_once(
            raw_totals, dataclasses.replace(cfg, di_cap_fraction=wider)
        )
        # Only valid while the cohort still has no HD; if widening DI somehow
        # produced one, the ordinary caps apply.
        if w_report.hd_count == 0:
            w_report.di_cap_widened = True
            return w_final, w_report
    return final, report


def _calibrate_once(
    raw_totals: Sequence[float],
    cfg: CalibrationConfig,
) -> tuple[List[float], CalibrationReport]:
    n = len(raw_totals)
    if n == 0:
        return [], CalibrationReport(0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0)

    target_mean = (cfg.target_mean_low + cfg.target_mean_high) / 2.0

    # Rank positions by raw total (desc); the tie-break on -index keeps it stable.
    order = sorted(range(n), key=lambda i: (raw_totals[i], -i), reverse=True)
    sorted_raw = [float(raw_totals[i]) for i in order]

    # The epsilon stops float error flooring a whole place away (0.29 * 100 is
    # 28.999999999999996).
    n_hd = math.floor(cfg.hd_cap_fraction * n + 1e-9)
    n_di = math.floor(cfg.di_cap_fraction * n + 1e-9)

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
        # A guard of exactly 1.0 would snap scores onto T-1 — the very 59/69/79
        # this feature exists to eliminate — so the effective guard never drops
        # below 1.5 (the snap target is then 68.5 etc., clearly inside the band).
        g = max(cfg.boundary_guard, 1.5)
        for threshold in boundary_thresholds:
            if threshold - g <= s < threshold:
                return threshold - g
        return s

    def scores_for_shift(t: float) -> List[float]:
        # clamp -> avoid_boundaries -> round, all monotonic, so ranking and the
        # caps are preserved and the cohort mean is non-decreasing in t.
        # Rounding happens HERE (not after the cap checks) so a 69.96 can never
        # round up to 70.0 and slip into a band after the caps were verified.
        return [
            round(avoid_boundaries(_clamp(sorted_raw[p] + t, floor, ceil_by_rank[p])), 1)
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
        t0 = (lo + hi) / 2.0
        m0 = mean_for(t0)
        if cfg.target_mean_low <= m0 <= cfg.target_mean_high:
            return t0
        return _refine_shift(t0)

    def _refine_shift(t0: float) -> float:
        # With boundary avoidance on, mean_for is a STEP function: whole
        # clusters sit snapped at (threshold - guard) and then cross the
        # threshold together, so the mean jumps. The binary search converges
        # into such a jump and (lo+hi)/2 can land on the wrong side of it —
        # a mean outside the target band even though a better shift exists.
        # Scan the neighbourhood (the jump is at most ~guard wide) and keep the
        # shift whose mean is closest to the target, preferring in-band means.
        # Any t keeps the caps/floor/ceilings/order (they are structural), so
        # this only ever improves the mean.
        span = max(cfg.boundary_guard, 1.5) + 1.0

        def key(t: float) -> tuple:
            m = mean_for(t)
            in_band = cfg.target_mean_low <= m <= cfg.target_mean_high
            return (0 if in_band else 1, abs(m - target_mean))

        best_t, best_key = t0, key(t0)
        steps = int(span * 2 / 0.05)
        for j in range(steps + 1):
            cand = t0 - span + j * 0.05
            k = key(cand)
            if k < best_key:
                best_t, best_key = cand, k
        return best_t

    t = solve_shift()
    scores_sorted = scores_for_shift(t)

    # DI-overflow correction. Scores are non-increasing in rank, so the ranks
    # occupying the DI band form a contiguous run; demote the ceiling of the
    # ENTIRE SUFFIX from that run's last member downward. A suffix demotion
    # keeps the ceiling sequence non-increasing, which is what guarantees the
    # clamp can never invert the ranking (demoting a single mid-sequence rank
    # could put a higher-raw student below a lower-raw one). Each pass moves
    # the demotion frontier strictly upward, so this terminates.
    for _ in range(n + 1):
        di_positions = [
            p for p in range(n) if cfg.di_lower <= scores_sorted[p] < cfg.di_upper
        ]
        if len(di_positions) <= n_di:
            break
        frontier = max(di_positions)  # last rank currently inside the DI band
        for p in range(frontier, n):
            ceil_by_rank[p] = min(ceil_by_rank[p], cfg.base_ceiling)
        t = solve_shift()
        scores_sorted = scores_for_shift(t)

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
    ``[0, max_score]``; clamping drift is then spread across the components
    with remaining headroom so the sum matches ``new_total`` exactly (clamped
    to what the components can hold, ``[0, sum(max_scores)]``). Inputs are
    clamped first — and non-finite values (NaN/inf) zeroed — so impossible
    model output cannot propagate into the exported grades.
    """
    def _finite(x: float) -> float:
        x = float(x)
        return x if math.isfinite(x) else 0.0

    maxes = [max(0.0, _finite(m)) for m in max_scores]
    scores = [_clamp(_finite(s), 0.0, maxes[i]) for i, s in enumerate(scores)]
    n = len(scores)
    if n == 0:
        return []
    # The components can only ever hold [0, sum(maxes)] — an impossible target
    # is clamped so no component leaves its own range.
    target = _clamp(_finite(new_total), 0.0, sum(maxes))
    old_total = _finite(old_total)

    if old_total <= 0:
        total_max = sum(maxes) or 1.0
        out = [round(_clamp(target * m / total_max, 0.0, m), 1) for m in maxes]
    else:
        factor = target / old_total
        out = [round(_clamp(scores[i] * factor, 0.0, maxes[i]), 1) for i in range(n)]

    # Distribute the remaining drift across components, most headroom first.
    # A single component may not have enough headroom to absorb it all (e.g. a
    # dominant component already at its max when the report is curved up), so
    # keep spreading until the sum matches the target. Because target lies in
    # [0, sum(maxes)], the cohort as a whole always has enough headroom and the
    # loop terminates within n passes.
    for _ in range(n + 1):
        drift = round(target - sum(out), 1)
        if abs(drift) < 0.1:
            break
        if drift > 0:
            order = sorted(range(n), key=lambda i: maxes[i] - out[i], reverse=True)
        else:
            order = sorted(range(n), key=lambda i: out[i], reverse=True)
        moved = False
        for i in order:
            room = (maxes[i] - out[i]) if drift > 0 else out[i]
            if room < 0.05:
                continue
            step = math.copysign(min(abs(drift), room), drift)
            out[i] = round(_clamp(out[i] + step, 0.0, maxes[i]), 1)
            moved = True
            drift = round(target - sum(out), 1)
            if abs(drift) < 0.1:
                break
        if not moved:
            break
    return out
