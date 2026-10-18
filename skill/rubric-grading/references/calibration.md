# Calibration: rules, options, pitfalls

## What the engine does (standard mode)
1. Ranks students by raw total; the ranking is never changed.
2. Gives ranks score ceilings so the caps hold by construction: the top
   floor(HD% x N) may reach 95, the next floor(DI% x N) may reach 79, everyone
   else is capped at 69. Everyone has a soft floor of 50. Equal raw totals
   share a ceiling: a tie straddling a cap goes to the lower band together
   (the cap is a hard maximum; input order never decides).
3. Adds one global shift to every raw score and binary-searches it so the
   cohort mean hits the target band (target +/- 0.5).
4. With boundary avoidance on, a score in [T - buffer, T) for T in 60/70/80
   is pushed to T - buffer (58.5 / 68.5 / 78.5 with the 1.5 buffer; the
   buffer is never below 1.5 so nobody lands on exactly 59/69/79).
5. If the cohort ends up with no HD (the HD cap rounds to 0 places, i.e.
   N < 10 at 10%, or nobody reaches 80), it re-runs with the DI cap at the
   "no HD" fraction (default 30%).
6. Component scores are then rescaled to each final total, spreading any
   clamping across components, so the printed table always adds up.

## Options professors ask for
- **Exempt weak papers from the floor**: `--no-floor s1,s2` - they get raw
  + the cohort shift. Recommend this when a paper is missing most required
  components; lifting it to 50 gives it the same Pass as students who did
  the work. Tell the professor the mean will drop (by roughly
  (50 - their marks) / N) and ask whether to recalibrate the rest.
- **Pin a mark**: `--fixed s1=50,s2=35.2` (e.g. "raise X to 50 and Y by the
  same amount"). Pinned students are outside calibration: the target mean and
  caps apply to everyone else, then the pinned marks are inserted unchanged
  (the summary's final_mean and band counts exclude them). The whole-class
  mean therefore moves with the pins - tell the professor.
- **Raise/lower the mean without changing anyone's band**:
  `--preserve-bands WORK/finals_previous.json --mean 67` - everyone shifts
  together but stays inside their previous band (capped at 1.5 below the next
  cut-off). Re-running the standard engine at a higher mean would instead
  push some students across a band edge.
- **Different caps/mean**: `--mean`, `--hd`, `--di`, `--di-no-hd`,
  `--guard`, `--no-avoid`.

## Tell the professor about the side effects
- **Quota cliff**: with a 20% DI cap, the last student inside the quota and
  the first outside may be 1 raw mark apart but ~10 final marks apart
  (78.5 vs 68.5). Name the students at the cliff.
- **Ties at a cap**: students whose equal raw total straddled a cap were all
  held at the lower band; the summary lists them in `cap_ties` (the Word
  summary too). Name them so the professor can review.
- **Clusters**: several students on 78.5 / 68.5 / 58.5 is expected (ceilings
  plus boundary avoidance). The ranking is still preserved.
- **Floor lifts**: list every paper lifted to 50 with its raw mark and a
  recommendation.
- **Low differentiation**: if raw scores barely spread (sd < ~6), the curve
  can only rank what the grading produced - say so.
- Always check the summary the script prints: final_mean inside the target,
  band counts within caps, rank_inversions and boundary_scores empty.

## Adding a late paper to an already-published cohort
Do not recalibrate the published students unless the professor wants it.
Place the new paper on the existing curve: final = raw + published shift,
then apply the same ceilings (if the DI quota is already full, cap at 68.5)
and boundary rule. Recompute the cohort mean and confirm it is still in the
target band. Offer the full recalibration as an alternative, showing how each
published mark would move.
