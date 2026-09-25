# Per-student grader instructions

Fill in WORK and SID, then give this to one grader (subagent or yourself).

---

Grade ONE student submission.

1. Read the grading brief: WORK/brief.md (rubric, required elements,
   professor's policies, how this marker marks, anchor papers with the
   professor's ACTUAL marks, feedback rules).
2. Read the task brief if provided: WORK/task.txt.
3. Read everything in WORK/packs/SID/ - MANIFEST.txt, the full report (in
   chunks if long), every workbook extract, slide text. Cross-check the key
   numbers the report states against the spreadsheet (e.g. VaR, beta,
   expected return, hedge sizes, totals, P&L). Record matches and
   contradictions with both numbers.
   - Workbook extracts are trimmed (60 rows x 40 columns per sheet): do not
     claim something is absent from the spreadsheet because it is not in the
     trimmed rows.
   - Embedded images are invisible: never call a figure/screenshot missing.
4. Score each criterion on the professor's scale, placing this paper against
   the anchors criterion by criterion ("closer to the DI or the CR anchor
   here, because..."). Use the professor's decimal style (21, 22.5, 6.8).
   Be discriminating - use the full range the evidence supports; do not
   default to the middle.
5. Write the notes and overall comment per the feedback rules in the brief.
6. Save JSON to WORK/results/SID.json:
   {"sid": "SID", "student_name": "<from cover page or empty>",
    "display_name": "<Name (SID)>",
    "components": [{"name": "<exact rubric name>", "max": <max>, "score": <x>,
                    "notes": ["..."]}, ... one per criterion, rubric order],
    "total": <sum of scores>, "overall_comment": "...",
    "evidence": {"...key facts...": "...", "issues": ["..."]}}
7. Validate with python: loads, one entry per criterion, 0 <= score <= max,
   total equals the sum (within 0.05), every string pure ASCII.
   Reply with one line: SID, total, the criterion scores.
