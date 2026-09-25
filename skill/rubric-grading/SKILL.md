---
name: rubric-grading
description: Grade a whole cohort of university assignments against a rubric, calibrated to the professor's own marked reference papers, and export a Word grade report with per-criterion marks and student feedback. Use this whenever a lecturer/professor/tutor wants student reports, essays, trading or investment reports, group assignments or Excel-backed submissions marked, curved/moderated, "calibrated" to a target mean or HD/DI caps, re-calibrated after a change, or wants feedback comments written for students - even if they just say "chấm bài", "mark these", "grade my class", "cham giup toi", or upload a batch of submissions with a rubric. Also use it for replying to a student's marking appeal/reconsideration request about a mark produced this way.
---

# Rubric grading with cohort calibration

This skill reproduces a professor's marking of a whole class: read the task
and rubric, learn the professor's standard from reference papers they already
marked, grade every submission on that scale, calibrate the cohort to the
professor's distribution rules, and deliver a Word report with marks and
feedback they can paste into the LMS.

The professor is the authority. Your job is consistency with *their* marking,
not an abstract ideal: when their anchors and your instinct disagree, the
anchors win (see "Learn how this marker marks").

## Workflow

1. **Collect the setup** - task brief, rubric (criteria, maxima, band
   descriptors), and reference papers with the professor's ACTUAL marks per
   criterion (ideally one per band: HD / DI / CR / PASS). Ask for the marks if
   only the papers are given; band labels alone are a weak anchor. Also ask
   (or confirm defaults) for the calibration policy - see step 6.
2. **Intake submissions** as they arrive, batch by batch. Do not start grading
   until the professor says so (they often upload in many rounds).
3. **Profile each anchor** into a short written profile.
4. **Write the grading brief** that every grader will share.
5. **Grade each student** - one independent grader per submission.
6. **Calibrate the cohort** with `scripts/calibrate.py`.
7. **Export** with `scripts/export_docx.py` and report back with flags.

Install dependencies once: `pip install -r scripts/requirements.txt`.
Keep all working files in a scratch/work directory, not in the user's repo -
they contain student names, IDs and marks.

### 2. Intake

```bash
python scripts/intake.py --work WORK path/to/upload1.docx path/to/upload1.xlsx ...
```

Run it on every new batch; it merges into `WORK/registry.json`. It groups
files by student ID (in the filename, else in the text), falls back to shared
filename stems for ID-less files, and builds `WORK/packs/<id>/` (report text,
curated workbook extracts, slide text). After each batch, show the professor a
short table: student, files matched, and flags - missing spreadsheet, missing
report, unreadable file, UNMATCHED files - and a running "X/N received".
Resolve UNMATCHED files by opening their cover page (name/ID) and moving them
by hand in the registry. Files the professor sends outside the batch (e.g. a
chart image) belong to a student - ask whose, then note it in that pack.

### 3. Anchor profiles

Read each anchor paper *and its spreadsheet*. For each, write
`WORK/anchor_<BAND>.md`: what it does per criterion, the errors it still
contains (HD papers are rarely error-free - record that so graders do not
demand perfection for HD), and what kept it below the next band. Copy the
professor's per-criterion marks and their comment verbatim into
`WORK/anchor_marks.md`.

**Sanity-check every anchor.** If a paper's visible quality contradicts its
mark (e.g. a "PASS" that reads like a DI), stop and ask the professor why,
before grading anything. In one real case the PASS paper was polished prose
that lacked objectives, selection criteria, per-stock evidence, derivative
quote tables and an active-portfolio section - the professor marks *coverage
and evidence of required elements*, not prose quality. That lesson changed
every grade in the cohort; guessing it wrong would have inflated the class.

### 4. The grading brief

Write `WORK/brief.md` containing: the rubric (exact criterion names, maxima,
band thresholds), the task's required elements as a checklist, the
professor's policies (word limits, late work, missing files), "how this
marker marks" (the lessons from step 3), the four anchor profiles with marks,
and the feedback rules from `references/feedback_style.md`. Every grader reads
this file, so the whole cohort is judged by one standard.

### 5. Grade each student

Use one fresh subagent per student when subagents are available (they run in
parallel and each gets clean context - important for consistency and for
fitting long reports); otherwise grade one at a time yourself, re-reading the
brief for each. Give each grader `references/grader_prompt.md` with the
student ID filled in. Each writes `WORK/results/<id>.json`:

```json
{"sid": "s1234567", "student_name": "", "display_name": "Name (s1234567)",
 "components": [{"name": "<exact rubric name>", "max": 30, "score": 21.5, "notes": ["..."]}],
 "total": 0.0, "overall_comment": "...", "evidence": {"issues": ["..."]}}
```

Scores are on the professor's raw scale (the anchors' scale). Check outliers
yourself: very low totals, a 0 on any criterion, anything that contradicts
the grader's own evidence. Grep the report to confirm a claimed absence
(e.g. "no ethics section") before accepting it.

### 6. Calibrate

Default policy (confirm with the professor; they may change any of it):
target mean 65.5, HD (>=80) cap 10% of cohort, DI (70-79) cap 20%, DI cap
widens to 30% when the cohort has no HD, avoid 59/69/79 (buffer 1.5 -> scores
land on 58.5/68.5/78.5), soft floor 50 (minimise fails).

```bash
python scripts/calibrate.py --results WORK/results --out WORK/finals.json \
    --mean 65.5 --hd 10 --di 20 --di-no-hd 30 --guard 1.5
```

Read `references/calibration.md` before running it - it explains the
options professors actually ask for (exempt a paper from the floor, pin a
mark, raise the mean without changing anyone's band) and the side effects to
tell them about (clusters at 78.5/68.5, cliffs at the quota edge, papers
lifted to the floor).

### 7. Export and report back

```bash
python scripts/export_docx.py --results WORK/results --finals WORK/finals.json \
    --title "COURSE - Assessment N" --out grades_COURSE_AN.docx --names s123="Name"
```

Send the .docx to the professor. In your message: the band distribution, the
table of raw -> final per student, and a short list of things they must
decide or check - papers lifted to the floor (recommend whether they deserve
it), quota cliffs, zeros on a criterion, missing required files, late
evidence, unnamed students. Never commit grade files to a git repository.

## Appeals and reconsideration requests

When a student disputes a mark, read `references/appeals.md`. In short:
verify every claim in their request against the submission, decide per
criterion using the rubric descriptors and cohort consistency (not the
descriptive words in the feedback), check for clerical errors first, and
draft a reply that describes rather than rates.

## Files

- `scripts/intake.py`, `scripts/calibrate.py`, `scripts/export_docx.py` - see above.
- `scripts/grading_lib/` - parser, calibration engine and Word exporter
  vendored from the professor's grading app (tested: encodings, broken docx,
  zips, embedded-image notes, rank-preserving caps).
- `references/feedback_style.md` - how notes and comments must be written.
- `references/grader_prompt.md` - the per-student grader instructions.
- `references/calibration.md` - calibration rules, options and pitfalls.
- `references/appeals.md` - handling a marking appeal.
