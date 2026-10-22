---
name: student-email-reply
description: Find a student's email (marking appeal, grade clarification, reconsideration request, extension or feedback question) in the lecturer's Gmail, look up that student's marks and feedback, check their claims against the rubric and their submission, and save a short, teacher-style reply as a Gmail DRAFT for the lecturer to review - never sent. Use this whenever a lecturer/professor/tutor says "reply to this student", "draft a response to her email", "tra loi email sinh vien", "phan hoi khieu nai diem", "find the result of sXXXXXXX and reply", pastes a student's appeal, or forwards one - even if they only give a student ID or email address.
---

# Student email reply (lecturer voice, Gmail draft)

The lecturer wants a reply they can send with minimal editing: short, polite,
firm, specific, and correct. A long point-by-point rebuttal reads like a
legal brief and invites a longer reply back - a lecturer who saw a 900-word
draft asked for it to be cut to ~280 words "like a teacher replying to a
student". Aim for that: 180-320 words, 4-5 short paragraphs.

## Workflow

1. **Find the email.** Search Gmail for the student ID, email address or
   name (`s4143428`, `from:...`, `subject:(assessment OR grade OR mark)`).
   Students often email the course coordinator, who forwards it - read the
   whole thread and note who the student actually wrote to (that person
   signs the reply). If several emails match, ask which one.
2. **Find the result.** In order: the grade report the lecturer gave you or
   produced earlier in the conversation (e.g. a `grades_*.docx` from the
   rubric-grading skill), the lecturer's Drive/Gmail attachments, or ask the
   lecturer for the marks per criterion and the feedback issued. Never
   invent marks. If you had to reconstruct marks (e.g. rescaling stored raw
   scores), say so and ask the lecturer to confirm against the LMS.
3. **Check before deciding** (see `references/checks.md`):
   - Clerical errors first: does the mark the student quotes match the
     record on the same scale? If the record is wrong, the reply corrects
     it in the student's favour.
   - Verify each claim against the submission if you have it. Students
     often concede the real issues inside the appeal - those concessions
     are usually the strongest basis for the decision.
   - Find the ONE or TWO issues that actually set the band (often a missing
     required component, or errors in the core calculations), rather than
     answering every paragraph.
   - Decide: marks stand / correct an error / adjust. Changing a mark is the
     lecturer's call - if you think one should change, say so to the
     lecturer in chat and draft both versions only if asked.
4. **Draft the reply** in the student's language (usually English) using
   `references/template.md`. Answer any direct technical question the
   student asks, briefly - it shows the email was read.
5. **Save as a Gmail draft** with `create_draft` (to: the student's address,
   subject: "Re: <their subject>"). Never send. If an earlier draft exists
   for the same student, update it (`update_draft`) or tell the lecturer.
6. **Report back** in the lecturer's language: the student's result, the
   decision and the 2-3 reasons, the draft link, and a short checklist:
   confirm marks against the LMS, fill placeholders (review-process link),
   and send from the account the student wrote to (a draft in a personal
   Gmail should be copied into the official account and sent as a reply in
   the original thread).

## Writing rules
- Warm opening, one line: thank them and acknowledge the care taken.
- Explain the decision through the rubric and the brief's required
  components - not through descriptive words in earlier feedback (a note
  that describes content does not denote a band).
- Describe, don't rate: no new praise words (excellent, strong, very good)
  that could be quoted in a further appeal. Critical points must be
  specific (what, where) - not "lacks depth".
- Never mention AI tools or graders, calibration, curves, caps, raw scores,
  or other students. If moderation is relevant: "all submissions were
  reviewed together to apply the same standard".
- Do not list every mark again unless one changed; the student has them.
- Close with: the outcome in one sentence, an offer to meet, and the formal
  review pathway as a placeholder `[formal review process]`.
- Sign with the name of the person the student wrote to (e.g. "Long Do").
- Plain text, no markdown, no bullet symbols.
