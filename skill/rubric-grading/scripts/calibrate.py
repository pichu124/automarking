"""Cohort calibration of per-student grade JSONs.

Usage (standard two-pass curve):
    python calibrate.py --results RESULTS_DIR --out finals.json \
        [--mean 65.5] [--hd 10] [--di 20] [--di-no-hd 30] [--guard 1.5] [--no-avoid] \
        [--no-floor s123,s456] [--fixed s123=50,s456=35.2]

Usage (band-preserving lift of an already-calibrated cohort, e.g. "raise the
mean to 67 without changing anyone's band"):
    python calibrate.py --results RESULTS_DIR --out finals.json \
        --preserve-bands previous_finals.json --mean 67 [--fixed ...] [--guard 1.5]

Each RESULTS_DIR/*.json is one student: {"sid", "total", "components": [...]}.
Totals must be on a 0-100 scale (component maxima summing to 100); otherwise,
or for an unknown id in --fixed/--no-floor, it exits without writing anything.
Writes {"finals": {sid: final}, "summary": {...}} and prints a table.

Standard mode = the app's engine: rank-preserving curve to the target mean,
HD cap (score >= 80) and DI cap (70-79) as % of cohort, soft floor at 50, and
optional avoidance of 59/69/79 (scores in [T-guard, T) pushed to T-guard).
If the cohort ends up with no HD, the DI cap widens to --di-no-hd.
--no-floor: these students keep raw + the cohort shift (within 0-100) instead of the floor.
--fixed:    pin exact finals (they still count toward the cohort mean).
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grading_lib.calibration import calibrate_scores  # noqa: E402
from grading_lib.config import DEFAULT_CALIBRATION  # noqa: E402


def _kv(s: str) -> dict:
    out = {}
    for part in filter(None, (s or "").split(",")):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def _band_range(v: float, guard: float) -> tuple[float, float]:
    if v >= 80:
        return 80.0, 95.0
    if v >= 70:
        return 70.0, 80.0 - guard
    if v >= 60:
        return 60.0, 70.0 - guard
    if v >= 50:
        return 50.0, 60.0 - guard
    return 0.0, 50.0 - 0.1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mean", type=float, default=65.5)
    ap.add_argument("--hd", type=float, default=10)
    ap.add_argument("--di", type=float, default=20)
    ap.add_argument("--di-no-hd", type=float, default=30)
    ap.add_argument("--guard", type=float, default=1.5)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--no-floor", default="")
    ap.add_argument("--fixed", default="")
    ap.add_argument("--preserve-bands", default="")
    a = ap.parse_args()

    R = {}
    for p in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        R[r["sid"]] = r
    maxima = {s: sum(float(c["max"]) for c in r["components"]) for s, r in R.items()}
    off_scale = [f"{s} ({m:g})" for s, m in maxima.items() if abs(m - 100) > 0.5]
    if off_scale:
        sys.exit(
            "calibrate.py: totals must be on a 0-100 scale - each result's component maxima "
            f"must sum to 100, but these do not: {', '.join(off_scale)}. Convert those results "
            "to a 0-100 scale and rerun. Nothing was written."
        )
    raw = {s: float(r["total"]) for s, r in R.items()}
    fixed = _kv(a.fixed)
    no_floor = {s.strip() for s in a.no_floor.split(",") if s.strip()}
    unknown = sorted((set(fixed) | no_floor) - set(raw))
    if unknown:
        sys.exit(
            f"calibrate.py: unknown student id(s) in --fixed/--no-floor: {', '.join(unknown)} "
            f"(no such sid in {a.results}). Nothing was written."
        )
    n = len(raw)
    guard = max(a.guard, 1.5)

    if a.preserve_bands:
        prev_run = json.load(open(a.preserve_bands, encoding="utf-8"))
        prev = prev_run.get("finals", prev_run)
        # The bands were set under the previous run's caps.
        diag = {k: v for k, v in prev_run.get("summary", {}).items() if k in ("hd_cap", "di_cap")}
        ids = [s for s in raw if s not in fixed]
        missing = [s for s in ids if s not in prev]
        if missing:
            sys.exit(
                f"calibrate.py: {a.preserve_bands} has no previous final for {', '.join(missing)}; "
                "pin them with --fixed or rerun the standard calibration. Nothing was written."
            )

        def finals_for(t):
            out = {}
            for s in ids:
                lo, hi = _band_range(prev[s], guard)
                out[s] = round(min(hi, max(lo, raw[s] + t)), 1)
            return out

        lo, hi = -40.0, 40.0
        for _ in range(100):
            mid = (lo + hi) / 2
            m = (sum(finals_for(mid).values()) + sum(fixed.values())) / n
            lo, hi = (mid, hi) if m < a.mean else (lo, mid)
        shift = (lo + hi) / 2
        finals = finals_for(shift)
        finals.update(fixed)
        mode, widened = "band-preserving", False
    else:
        cfg = dataclasses.replace(
            DEFAULT_CALIBRATION,
            target_mean_low=a.mean - 0.5, target_mean_high=a.mean + 0.5,
            hd_cap_fraction=a.hd / 100, di_cap_fraction=a.di / 100,
            di_cap_fraction_no_hd=a.di_no_hd / 100,
            avoid_band_boundaries=not a.no_avoid, boundary_guard=guard,
        )
        ids = list(raw)
        fin, rep = calibrate_scores([raw[s] for s in ids], cfg)
        finals = dict(zip(ids, fin))
        shift = rep.shift
        for s in no_floor:
            finals[s] = round(min(100.0, max(0.0, raw[s] + shift)), 1)
        finals.update(fixed)
        mode, widened = "standard", rep.di_cap_widened
        diag = {"hd_cap": rep.hd_cap, "di_cap": rep.di_cap, "raw_std": rep.raw_std,
                "low_discrimination": rep.low_discrimination}

    vals = list(finals.values())
    order = sorted(raw, key=lambda s: -raw[s])
    inversions = [(x, y) for x, y in zip(order, order[1:]) if finals[x] < finals[y] - 0.01 and x not in fixed and y not in fixed]
    boundary = [s for s, v in finals.items() for T in (60, 70, 80) if not a.no_avoid and T - guard < v < T]
    summary = {
        "mode": mode, "n": n, "raw_mean": round(statistics.mean(raw.values()), 2),
        "final_mean": round(statistics.mean(vals), 2), "shift": round(shift, 2),
        "hd": sum(v >= 80 for v in vals), "di": sum(70 <= v < 80 for v in vals),
        "cr": sum(60 <= v < 70 for v in vals), "pa": sum(50 <= v < 60 for v in vals),
        "fail": sum(v < 50 for v in vals), "di_cap_widened": widened, **diag,
        "rank_inversions": inversions, "boundary_scores": boundary,
    }
    json.dump({"finals": finals, "summary": summary}, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    for s in sorted(finals, key=lambda s: -finals[s]):
        print(f"{s:10s} raw {raw[s]:5.1f} -> {finals[s]:5.1f}")


if __name__ == "__main__":
    main()
