# 🎓 AI Grading Studio

A course-agnostic, rubric-aligned grading web app for university professors. The
professor uploads the **task**, the **rubric**, and **four graded reference
samples**, then batch-uploads a whole cohort of reports. The app grades each
report against the rubric with Claude, then applies a **Two-Pass statistical
calibration engine** so the final cohort grades obey a target distribution, and
exports a structured **Microsoft Word** grade report.

---

## 1. Architecture

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | **Streamlit** | Fast to build, native file-upload + download widgets |
| LLM | **Pluggable** — Claude, Gemini, Groq, Ollama, OpenAI, … | Pick by cost/quality; see Providers below |
| Structured output | Claude: `messages.parse` + Pydantic. Others: JSON mode + robust parse | Validated objects, never brittle text parsing |
| Doc parsing | `python-docx`, `pypdf`, `zipfile` | DOCX / PDF / TXT / ZIP ingestion |
| Export | `python-docx` | The rubric-shaped `.docx` deliverable |
| Calibration | Pure-Python, unit-tested | Deterministic, auditable, no LLM in the loop |

### Providers (free & paid)

Anthropic's API is paid, so the app ships a **provider selector** (sidebar). Most
providers speak the OpenAI Chat Completions protocol, so one adapter
(`OpenAICompatibleProvider`) covers many back-ends; the native Anthropic path
keeps Claude's prompt caching. Switch any time — the rest of the pipeline is
unchanged.

| Provider | Cost | Notes |
|---|---|---|
| **Anthropic Claude** | Paid | Best grading quality; prompt caching |
| **Google Gemini** | 🆓 **free tier** | `gemini-2.0-flash`; key at aistudio.google.com |
| **Groq** | 🆓 **free tier** | Fast, open models (Llama 3.3 70B); console.groq.com |
| **Ollama** | 🆓 **free, local** | Runs offline on your machine; no API key, full privacy |
| **OpenRouter** | Mixed | Aggregator; some `:free` models |
| **DeepSeek / Mistral / OpenAI** | Cheap–paid | OpenAI-compatible |
| **Custom** | — | Any OpenAI-compatible endpoint (your own base URL) |

**Recommendation:** for zero cost use **Gemini** or **Groq** (free tiers) for
convenience, or **Ollama** for fully-local/private grading. Use **Claude** when
grading quality and consistency matter most. Quality of feedback and calibration
fidelity track the underlying model — small local models are cheaper but blunter.
Add a provider in `src/providers.py` (`PROVIDER_PRESETS`).

### Running fully free & offline with Ollama

[Ollama](https://ollama.com) runs an open model on your own machine — no API key,
no cost, nothing leaves your computer. Setup:

1. **Pull a model:** `ollama pull llama3.1` (or a stronger one — see below).
2. **Raise the context window.** Ollama's default is ~4k tokens, far too small
   for a rubric + four samples + a report, so long inputs get silently cut off.
   Set `OLLAMA_CONTEXT_LENGTH` and restart Ollama:
   - **Windows (PowerShell):**
     ```powershell
     [System.Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH","32768","User")
     ```
     then quit Ollama from the system tray and reopen it.
   - **macOS/Linux:** `export OLLAMA_CONTEXT_LENGTH=32768` then `ollama serve`.
   - *Alternative (per-model):* a Modelfile —
     `FROM llama3.1` + `PARAMETER num_ctx 32768`, then
     `ollama create grading -f Modelfile`, and use model `grading` in the app.
3. **In the app sidebar:** pick **Ollama (local, free)**, set the model name to
   what you pulled, and set **📏 Document length** so a whole request fits your
   context (use the token count in the 📊 Estimate panel as a guide — request ≈
   static context + report + output).

**Model choice ↔ machine.** Bigger models grade better but need more RAM/VRAM:
`llama3.1:8b` / `qwen2.5:7b` run on ~16GB; `qwen2.5:14b` is noticeably sharper if
you have headroom. Local models give rougher feedback than Claude/Gemini and may
occasionally emit malformed JSON (those reports are recorded as errors and the
batch continues). For best local quality, prefer a 14B+ instruct model.

### Compact anchor profiles (don't truncate samples — distil them)

Trimming a reference sample to fit a small context means the model only sees its
opening. Instead, use **🧭 Read & compress samples** (after structuring the
rubric): the model reads each sample **in full, once**, and distils it into a
~150-word "anchor profile" capturing the whole paper's quality per rubric
component. Grading then anchors on the profiles — which cut the sample portion of
each request from ~24k tokens (four full papers) to ~500 tokens, so even an
8B model on Ollama keeps the full-sample calibration signal. Toggle it off to go
back to raw samples. Big-context models (Claude/Gemini) can skip it and read full
samples directly. Logic: `summarize_samples_with_fallback` in
`src/grading_engine.py`.

### Context size & free-tier limits

Each grading request bundles the task, rubric, four reference samples and one
student report. The app applies **generous length caps** as a *safety valve*
(`MAX_*_CHARS` in `src/config.py`) — **not** a routine trim. Defaults are sized
so a normal full assignment (≈2,500 words) and full reference papers are read in
their **entirety**; only pathologically large files (e.g. a 3.4MB PDF that
extracts to hundreds of KB) are shortened. When truncation would happen, the
"📊 Estimate" panel **warns you exactly which documents** are affected, so no
information is ever lost silently. You control it live: the sidebar **📏
Document length** sets the per-document cap, and **0 = read everything** (best
with large-context models like Claude or Gemini). Lower it only to fit tight
free tiers like Groq.

Provider notes:
- **Groq free tier ≈ 12k tokens/min** — too small for a full rubric + four
  samples. Use it only with small contexts; otherwise prefer the options below.
- **Gemini free tier / Claude** have very large context windows — best for full
  rubrics and samples. **Ollama** (local) has no token billing at all.
- The "📊 Estimate cost & time" panel shows the per-batch token count so you can
  see whether you'll hit a limit before running.

### Automatic fallback — *without breaking consistency*

You can enable an ordered list of **fallback providers** (sidebar → "Automatic
fallback"). The calibration only makes sense if the whole cohort is graded by
**one** model — different models score on different baselines, so a mixed cohort
can't be calibrated on a single scale. The fallback therefore works like this:

- **Transient rate-limits / 5xx** are absorbed *inside* a provider by the SDK's
  own exponential backoff (`max_retries=4`). Most 429s never cause a switch.
- **Only a persistent provider-health failure** (rate-limit after retries, auth
  error, server/connection error) triggers a switch — and switching **re-grades
  the entire cohort from scratch** with the next provider, discarding the failed
  provider's partial work.
- **Per-report content errors** (an unparseable document, invalid JSON for one
  report) do *not* switch provider; they're recorded and skipped.

The guarantee: the final cohort is always graded end-to-end by a single
provider. The UI and the `.docx` ("Graded by …") record which one, and the
result carries an audit trail of any fallback. Logic lives in
`grade_batch_with_fallback` (`src/grading_engine.py`); the consistency invariant
is covered by `tests/test_fallback.py`.

### Workflow

```
New Session ─▶ Upload Task + Rubric + 4 Reference Samples ─▶ Structure Rubric
     │                                                              │
     ▼                                                              ▼
Batch-upload all student reports  ──────────────▶  PASS 1: grade each report
(PDF / DOCX / ZIP)                                  independently vs. rubric
                                                    + reference anchors (Claude)
                                                              │
                                                              ▼
                                          PASS 2: cohort curve calibration
                                          (mean 65–66%, HD ≤10%, DI ≤20%,
                                           fails minimised)
                                                              │
                                                              ▼
                                          FINAL ADJUSTMENT: rescale each
                                          report's component scores to its
                                          calibrated total
                                                              │
                                                              ▼
                                          Structured .docx grade report
```

### The Two-Pass calibration engine (`src/calibration.py`)

The grader's job in Pass 1 is to get the **relative quality and the component
breakdown** right; absolute magnitudes are fixed in Pass 2. The calibrator uses
**banded rank mapping with mean targeting** to satisfy four hard constraints at
once:

1. **Rank** reports by raw total (descending). The LLM's ordering is treated as
   ground truth — calibration never reorders, only rescales.
2. **Band ceilings by rank** make the caps hold *by construction*:
   - top `floor(0.10·N)` ranks → ceiling 95 (may reach HD)
   - next `floor(0.20·N)` ranks → ceiling 79 (DI at most, never HD)
   - everyone else → ceiling 69 (never DI/HD)
   - a soft **floor of 50** for everyone — this is how "aggressively minimise
     fails" is implemented.
3. **Global additive shift + clamp.** One shift `t` is applied to every raw
   score, then clamped into each rank's `[floor, ceiling]`. Because the raw
   scores and the bounds are all non-increasing in rank, the clamp preserves
   ordering, and the cohort mean is monotonic and continuous in `t` — so a
   **binary search** lands the mean exactly inside `[65, 66]`.
4. **DI-overflow correction.** A top-band report whose calibrated score lands in
   `[70, 80)` still counts toward the DI cap. If the DI band overflows, the
   lowest-ranked DI member is demoted to ceiling 69 and the shift is re-solved.
   Each demotion permanently removes a rank from the DI-eligible set, so the
   loop terminates.

Then the **final adjustment** rescales each report's component scores
proportionally so they sum to the calibrated total (`rescale_components`).

The constraints are covered by unit tests (`tests/test_calibration.py`),
including the adversarial cases: a cohort the LLM graded all-high (must compress
down), all-low (must lift up and zero out fails), a compressed top band (DI
overflow), tiny cohorts, and identical scores.

### Pre-flight cost & time estimate

Before running a batch, the **Grade** screen shows a "📊 Estimate cost & time"
panel: estimated cost, time range, and token counts for the selected provider,
plus a **provider comparison table** (cheapest first) so you can pick the
best-value option. The Claude estimate accounts for prompt caching (the shared
context is billed once, then read at ~0.1× per later report). Estimates are
indicative — token counts use a local ~4-chars/token heuristic and the prices in
`src/config.py` (`PROVIDER_PRICING`) are approximate and editable. Logic and
tests: `src/estimator.py`, `tests/test_estimator.py`.

### Prompt caching & cost

On the **Claude** path the expensive static context — the persona/instructions
plus this session's task, rubric and four reference samples — is sent as
**cached system blocks** (`cache_control: ephemeral`). Only the per-student
report varies, so every report after the first reads that large prefix from
cache (~0.1× input cost). See `src/prompts.py` → `build_grading_system_blocks`.
OpenAI-compatible providers send the same context as a plain system message
(`build_grading_system_text`); some of them apply their own server-side caching.

---

## 2. Project layout

```
automarking/
├── app.py                  # Streamlit UI (sessions, uploads, batch, download)
├── src/
│   ├── config.py           # Model + calibration targets
│   ├── models.py           # Pydantic schemas (LLM contract + internal types)
│   ├── document_parser.py  # DOCX / PDF / TXT / ZIP → text
│   ├── prompts.py          # The Master grading prompt + context builders
│   ├── llm_client.py       # Anthropic SDK wrapper (parse + caching + thinking)
│   ├── providers.py        # Provider registry + OpenAI-compatible adapter
│   ├── estimator.py        # Pre-flight cost/time estimate + provider compare
│   ├── calibration.py      # The Two-Pass calibration engine
│   ├── grading_engine.py   # Orchestration of Pass 1 → Pass 2 → adjustment
│   └── docx_export.py      # Rubric-shaped .docx generation
└── tests/test_calibration.py
```

---

## 3. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add a key for whichever provider you'll use
streamlit run app.py
```

Then in the browser:

0. In the **sidebar**, pick a provider (e.g. *Google Gemini · 🆓* or *Ollama ·
   🆓*) and model. Paste a key, or set the matching env var in `.env`.
1. **New Grading Session** and name the assignment.
2. Upload the **Task**, the **Rubric**, and the four **reference samples**
   (High Distinction, Distinction, Average, Pass), then **Structure rubric**.
3. **Batch-upload** all student reports (or one `.zip`).
4. **Run Two-Pass Grading** — review the calibrated cohort.
5. **Download** the `.docx` grade report.

Run the tests with:

```bash
python -m pytest tests/ -v
```

---

## 4. Output format

The generated `.docx` leads with a **cohort summary** (calibrated mean, HD/DI
counts vs. caps, fail count), then one page per student containing:

- **Student name / ID** (extracted from the report, falling back to filename)
- a **component table** that mirrors the uploaded rubric — `Component · Score ·
  Max · Professor's Notes` — with 2–3 punchy, technical bullets per component
- **Total / 100** and the grade band

Feedback is written in a strict **busy-academic persona** (see
`src/prompts.py`): direct, technical, no fluff, weaknesses first.

> **Note on fairness.** The calibration curve is a hard administrative
> distribution control. Professors should review the calibrated grades — the
> app surfaces both raw and final scores so any large adjustment is visible.
