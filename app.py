"""Streamlit UI for the AI grading application.

Workflow:
  1. New Grading Session — name it.
  2. Calibration data — upload Task, Rubric, and up to four reference samples
     (each slot accepts several files; any subset — or none — is enough to
     grade). Structuring the rubric turns it into weighted components.
  3. Batch upload — drop in all student reports (or a .zip).
  4. Grade — runs the Two-Pass engine and shows the calibrated cohort.
  5. Download — a structured .docx grade report.
"""
from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

import dataclasses

from src import config
from src.config import DEFAULT_CALIBRATION, DEFAULT_EFFORT, PROVIDER_PRICING

# Factory-default caps for the UI. Read from the immutable BASE_* constants:
# the sidebar mutates config.MAX_*, and app.py re-executes on every rerun, so
# capturing the mutable names here would leak one session's override into the
# next session's defaults.
_BASE_REPORT_CAP = config.BASE_MAX_REPORT_CHARS
_BASE_SAMPLE_CAP = config.BASE_MAX_SAMPLE_CHARS
from src.docx_export import build_grade_document, sanitize_text
from src.document_parser import (
    HEALTH_HINTS,
    ParsedDocument,
    group_by_student,
    parse_upload,
    read_failure_reason,
    text_health,
)
from src.estimator import compare_providers, estimate_batch, static_context_text
from src.grading_engine import (
    SAMPLE_LEVELS,
    GradingSession,
    ProviderSpec,
    grade_batch_with_fallback,
    recalibrate,
    structure_rubric_with_fallback,
    summarize_samples_with_fallback,
)
from src.providers import PROVIDER_PRESETS, preset_is_usable

load_dotenv()

st.set_page_config(page_title="AI Grading Studio", page_icon="🎓", layout="wide")

REFERENCE_KEYS = [
    ("HD", "High Distinction sample"),
    ("DI", "Distinction sample"),
    ("AVG", "Average sample"),
    ("PASS", "Pass sample"),
]


# --- Session state helpers -------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("session_name", "")
    st.session_state.setdefault("task_text", "")
    st.session_state.setdefault("rubric_text", "")
    st.session_state.setdefault("reference_samples", {})
    st.session_state.setdefault("components", None)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("batch_result", None)
    st.session_state.setdefault("marker_text", "")


def _first_text(uploaded) -> str:
    docs = parse_upload(uploaded)
    return docs[0].text if docs else ""


def _uploads_text(uploads) -> str:
    """Concatenate the parsed text of several uploads, labelled by filename."""
    docs = []
    for up in uploads:
        docs.extend(parse_upload(up))
    if len(docs) <= 1:
        return docs[0].text if docs else ""
    return "\n\n".join(f"=== File: {d.name} ===\n{d.text.strip()}" for d in docs)


def _calibration_config():
    """Build a CalibrationConfig from the sidebar controls (or the defaults)."""
    mid = st.session_state.get("cal_mean", 65.5)
    return dataclasses.replace(
        DEFAULT_CALIBRATION,
        target_mean_low=mid - 0.5,
        target_mean_high=mid + 0.5,
        hd_cap_fraction=st.session_state.get("cal_hd", 10) / 100.0,
        di_cap_fraction=st.session_state.get("cal_di", 20) / 100.0,
        avoid_band_boundaries=st.session_state.get("cal_avoid", True),
        boundary_guard=st.session_state.get("cal_guard", 1.5),
    )


def _read_uploaded_once(uploaded, text_key: str, file_id_key: str) -> None:
    """Read an uploaded file into session state exactly once per upload.

    Reading only when the file id changes means a later manual paste into the
    text area is not clobbered on every Streamlit rerun. Removing the file
    clears the text again — unless the user edited it by hand since.
    """
    shadow_key = file_id_key + "_shadow"
    if uploaded is None:
        if st.session_state.pop(file_id_key, None) is not None:
            if st.session_state.get(text_key, "") == st.session_state.get(shadow_key, ""):
                st.session_state[text_key] = ""
            st.session_state.pop(shadow_key, None)
        return
    if st.session_state.get(file_id_key) != uploaded.file_id:
        text = _first_text(uploaded)
        st.session_state[text_key] = text
        st.session_state[shadow_key] = text
        st.session_state[file_id_key] = uploaded.file_id


def _read_uploads_once(uploads, text_key: str, ids_key: str) -> None:
    """Multi-file variant: concatenate every uploaded file's text.

    Re-reads only when the set of files changes (add/remove), so manual edits
    in the text area survive Streamlit reruns. With more than one document the
    pieces are labelled with their filename so the model can tell them apart.
    Removing every file clears the text again — unless the user has edited it
    by hand since the upload (their edits win; a shadow copy of the last
    upload-produced text tells the two apart).
    """
    shadow_key = ids_key + "_shadow"
    if not uploads:
        if st.session_state.pop(ids_key, None) is not None:
            if st.session_state.get(text_key, "") == st.session_state.get(shadow_key, ""):
                st.session_state[text_key] = ""
            st.session_state.pop(shadow_key, None)
        return
    ids = tuple(u.file_id for u in uploads)
    if st.session_state.get(ids_key) == ids:
        return
    text = _uploads_text(uploads)
    st.session_state[text_key] = text
    st.session_state[shadow_key] = text
    st.session_state[ids_key] = ids


def _read_sample_uploads_once(key: str, uploads) -> None:
    """Read one reference-sample slot (several files allowed) exactly once.

    A sample submission often spans two files (Word essay + Excel workbook) —
    they are concatenated with filename labels, like the task and batch
    uploaders. Removing every file from the slot clears the sample again.
    """
    samples = st.session_state["reference_samples"]
    ids_key = f"_ref_{key}_ids"
    if not uploads:
        if st.session_state.pop(ids_key, None) is not None:
            samples.pop(key, None)
        return
    ids = tuple(u.file_id for u in uploads)
    if st.session_state.get(ids_key) == ids:
        return
    samples[key] = _uploads_text(uploads)
    st.session_state[ids_key] = ids


def _provider_chain() -> list[ProviderSpec]:
    """Ordered provider list: the sidebar primary, then any enabled fallbacks.

    The whole cohort is graded by ONE provider; fallbacks only kick in if the
    one above them is unavailable, and switching re-grades from scratch — so
    scores stay consistent.
    """
    primary = ProviderSpec(
        provider_key=st.session_state.get("provider_key", "anthropic"),
        model=st.session_state.get("model") or None,
        api_key=st.session_state.get("api_key") or None,
        base_url=st.session_state.get("base_url") or None,
        effort=st.session_state.get("effort", DEFAULT_EFFORT),
        temperature=st.session_state.get("temperature", 0.2),
        label=PROVIDER_PRESETS[st.session_state.get("provider_key", "anthropic")].label,
    )
    chain = [primary]
    for key in st.session_state.get("fallback_keys", []):
        if key == primary.provider_key:
            continue
        preset = PROVIDER_PRESETS[key]
        # Fallbacks use preset defaults + env-var keys (resolved in build_provider).
        chain.append(
            ProviderSpec(
                provider_key=key,
                effort=DEFAULT_EFFORT,
                temperature=st.session_state.get("temperature", 0.2),
                label=preset.label,
            )
        )
    return chain


# --- UI sections -----------------------------------------------------------

@st.cache_resource
def _app_version() -> str:
    """Short git commit of the running code — so you can tell at a glance
    whether a `git pull` has actually taken effect (requires a full app
    restart, not just a browser Rerun)."""
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Provider")
        st.caption(f"App version: `{_app_version()}` — after `git pull`, stop the "
                   "app (Ctrl+C) and run it again for changes to load.")

        keys = list(PROVIDER_PRESETS.keys())

        def _fmt(k: str) -> str:
            p = PROVIDER_PRESETS[k]
            tag = " · 🆓" if p.free else ""
            return f"{p.label}{tag}"

        # NOTE (applies to every widget below): widgets that feed their own
        # return value back through `value=`/`index=` WITHOUT a key get their
        # widget ID derived from that value — so the ID changes after an edit
        # and the NEXT edit is silently dropped. Stable `key=` + seeded session
        # state is the correct pattern.
        st.session_state.setdefault("provider_key", "anthropic")
        provider_key = st.selectbox(
            "Grading engine", keys, format_func=_fmt, key="provider_key",
            help="Anthropic is paid. Gemini & Groq have free tiers; Ollama is "
                 "fully free and runs locally.",
        )
        preset = PROVIDER_PRESETS[provider_key]
        if preset.note:
            st.caption(preset.note)
        if provider_key == "ollama":
            st.info(
                "Ollama checklist:\n"
                "1. `ollama pull llama3.1` (or another model)\n"
                "2. Raise context: set `OLLAMA_CONTEXT_LENGTH` (e.g. 32768) and "
                "restart Ollama — otherwise long inputs are silently cut.\n"
                "3. Keep **📏 Document length** so a request fits your context "
                "(check the token count in 📊 Estimate)."
            )

        # Model — prefilled with the preset default, editable.
        if st.session_state.get("_model_for") != provider_key:
            st.session_state["model"] = preset.default_model
            st.session_state["_model_for"] = provider_key
        st.text_input("Model", key="model")

        # API key — defaults to the relevant env var; hidden input to override.
        env_key = os.getenv(preset.api_key_env) if preset.api_key_env else None
        if preset.api_key_optional:
            st.caption("No API key required for this provider.")
            st.session_state["api_key"] = ""
        elif env_key:
            st.success(f"Using {preset.api_key_env} from the environment.")
            st.session_state["api_key"] = ""
        else:
            label = f"{preset.api_key_env or 'API key'}"
            st.session_state.setdefault("api_key", "")
            st.text_input(
                label, type="password", key="api_key",
                help="Or set it in your environment / .env file.",
            )
            if preset.api_key_env and not st.session_state["api_key"]:
                st.warning(f"Set {preset.api_key_env} or paste a key above.")

        # Base URL — only for custom endpoints.
        if provider_key == "custom":
            st.session_state.setdefault("base_url", "")
            st.text_input(
                "Base URL", key="base_url",
                placeholder="https://your-endpoint/v1",
            )
        else:
            st.session_state["base_url"] = ""

        # Generation controls differ by kind.
        if preset.kind == "anthropic":
            st.session_state.setdefault("effort", DEFAULT_EFFORT)
            st.select_slider(
                "Effort", options=["low", "medium", "high", "max"], key="effort",
            )
        else:
            st.session_state.setdefault("temperature", 0.2)
            st.slider(
                "Temperature", 0.0, 1.0, step=0.05, key="temperature",
                help="Lower = more consistent grading.",
            )

        # --- Automatic fallback ------------------------------------------
        st.divider()
        st.subheader("🛟 Automatic fallback")
        st.caption(
            "If the provider above becomes unavailable (e.g. rate-limited), the "
            "app re-grades the **whole cohort** with the next available "
            "provider — so every report stays on one consistent scale."
        )
        candidates = [k for k in PROVIDER_PRESETS if k != provider_key and k != "custom"]
        usable = [k for k in candidates if preset_is_usable(k)[0]]
        unusable = [k for k in candidates if not preset_is_usable(k)[0]]
        st.session_state["fallback_keys"] = [
            k for k in st.session_state.get("fallback_keys", []) if k in usable
        ]
        st.multiselect(
            "Fallback providers (in order)",
            options=usable,
            key="fallback_keys",
            format_func=lambda k: PROVIDER_PRESETS[k].label
            + (" · 🆓" if PROVIDER_PRESETS[k].free else ""),
            help="Only providers with a key already in your environment (or "
                 "keyless, like Ollama) are listed. Fallbacks use their default "
                 "model. ⚠️ A different model means a different grading scale — "
                 "the cohort is re-graded entirely, never mixed.",
        )
        if unusable:
            st.caption(
                "Not available as fallback (no key set): "
                + ", ".join(PROVIDER_PRESETS[k].label for k in unusable)
            )
        chain_labels = [preset.label] + [
            PROVIDER_PRESETS[k].label for k in st.session_state["fallback_keys"]
        ]
        st.caption("Grading order: " + " → ".join(chain_labels))

        # --- How much of each document to send ---------------------------
        st.divider()
        st.subheader("📏 Document length")
        st.session_state.setdefault("report_cap", _BASE_REPORT_CAP)
        st.session_state.setdefault("sample_cap", _BASE_SAMPLE_CAP)
        report_cap = st.number_input(
            "Student report — max characters (0 = full)",
            min_value=0, max_value=400_000, step=2000, key="report_cap",
            help="The work being graded — keep this HIGH so the model reads the "
                 "whole report. A 2,500-word report is ~16,000 chars.",
        )
        sample_cap = st.number_input(
            "Each reference sample — max characters (0 = full)",
            min_value=0, max_value=400_000, step=2000, key="sample_cap",
            help="Samples are calibration anchors — an excerpt is enough. LOWER "
                 "this first to shrink each request (biggest lever for small "
                 "contexts like Ollama). Every provided sample is sent every "
                 "request.",
        )
        # Applied at runtime — prompts read these from config on each call.
        config.MAX_REPORT_CHARS = report_cap
        config.MAX_SAMPLE_CHARS = sample_cap
        st.caption(
            f"≈ {(report_cap // 4) if report_cap else 0:,} tok/report · "
            f"{(sample_cap // 4) if sample_cap else 0:,} tok/sample × 4. "
            "0 = read in full."
        )

        st.divider()
        st.subheader("Calibration targets")
        st.session_state.setdefault("cal_mean", 65.5)
        st.slider("Target mean (%)", 55.0, 75.0, step=0.5, key="cal_mean")
        st.session_state.setdefault("cal_hd", 10)
        st.slider(
            "HD cap (%)", 0, 30, step=1, key="cal_hd",
            help="Max % of the cohort scoring ≥ 80.",
        )
        st.session_state.setdefault("cal_di", 20)
        st.slider(
            "Distinction cap (%)", 0, 40, step=1, key="cal_di",
            help="Max % of the cohort scoring 70–79.",
        )
        st.session_state.setdefault("cal_avoid", True)
        st.checkbox("Avoid boundary scores (59/69/79)", key="cal_avoid")
        if st.session_state["cal_avoid"]:
            # Minimum 1.5: a buffer of 1.0 would push scores onto exactly
            # 59/69/79 — the very scores the feature avoids.
            if st.session_state.get("cal_guard", 1.5) < 1.5:
                st.session_state["cal_guard"] = 1.5
            st.session_state.setdefault("cal_guard", 1.5)
            st.slider(
                "Boundary buffer", 1.5, 3.0, step=0.5, key="cal_guard",
                help="How far below each cut-off to push borderline scores "
                     "(1.5 → top of Credit becomes 68.5).",
            )
        st.caption("Change these anytime, then use **🔄 Re-calibrate** on results "
                   "to apply them for free (no tokens).")

        st.divider()
        if st.button("🆕 New Grading Session", use_container_width=True):
            for key in (
                "session_name", "task_text", "rubric_text",
                "reference_samples", "components", "documents", "batch_result",
                "marker_text", "sample_profiles", "use_profiles", "profile_hashes",
                "_task_file_id", "_task_file_id_shadow",
                "_rubric_file_id", "_rubric_file_id_shadow",
                "_marker_file_id", "_marker_file_id_shadow",
                "_batch_ids", "_batch_raw", "_raw_file_count",
                *(f"_ref_{k}_ids" for k, _ in REFERENCE_KEYS),
            ):
                st.session_state.pop(key, None)
            # Bumping the nonce gives every file_uploader a fresh key, which is
            # the only way to actually CLEAR uploaded files from the widgets —
            # otherwise the old uploads re-parse right back into the session.
            st.session_state["_uploader_nonce"] = st.session_state.get("_uploader_nonce", 0) + 1
            _init_state()
            st.rerun()


def _up_key(name: str) -> str:
    """File-uploader widget key, versioned by the reset nonce — giving the
    uploaders a fresh key on 🆕 New Grading Session is the only way to
    actually clear their files."""
    return f"{name}_{st.session_state.get('_uploader_nonce', 0)}"


def calibration_data_ui() -> None:
    st.header("1 · Session & Calibration Data")
    st.text_input(
        "Session / assignment name",
        key="session_name",
        placeholder="e.g. ECON301 — Market Analysis Essay (S1 2026)",
    )

    col_task, col_rubric = st.columns(2)
    with col_task:
        st.subheader("📋 The Task")
        task_files = st.file_uploader(
            "Assignment requirements (you can add several files)",
            type=["pdf", "docx", "xlsx", "xlsb", "xls", "txt", "md", "csv"],
            key=_up_key("task_up"),
            accept_multiple_files=True,
        )
        _read_uploads_once(task_files, "task_text", "_task_file_id")
        st.text_area(
            "Task text (you can also paste/edit here)",
            key="task_text", height=180,
        )

    with col_rubric:
        st.subheader("📐 The Rubric")
        rubric_file = st.file_uploader(
            "Grading criteria & weightings",
            type=["pdf", "docx", "xlsx", "xlsb", "xls", "txt", "md", "csv"],
            key=_up_key("rubric_up"),
        )
        _read_uploaded_once(rubric_file, "rubric_text", "_rubric_file_id")
        if rubric_file is not None and not st.session_state["rubric_text"].strip():
            st.warning(
                "Couldn't extract text from this file (it may be scanned or laid "
                "out in a text box). Paste the rubric text below instead."
            )
        st.text_area(
            "Rubric text (you can also paste/edit here)",
            key="rubric_text", height=180,
        )

    st.subheader("🎯 Reference Samples (calibration anchors)")
    st.caption(
        "Optional — grading works with any subset of anchors (or none), but "
        "each one you provide sharpens the standard. A slot accepts several "
        "files (e.g. a sample's Word report **and** its Excel workbook)."
    )
    cols = st.columns(4)
    samples = st.session_state["reference_samples"]
    for (key, label), col in zip(REFERENCE_KEYS, cols):
        with col:
            ups = st.file_uploader(
                label,
                type=["pdf", "docx", "xlsx", "xlsb", "xls", "txt", "md", "csv"],
                key=_up_key(f"ref_{key}"),
                accept_multiple_files=True,
            )
            _read_sample_uploads_once(key, ups)
            sample_text = samples.get(key, "")
            if not sample_text:
                st.caption("— empty")
            elif text_health(sample_text):
                st.caption(f"⚠️ loaded but {HEALTH_HINTS[text_health(sample_text)]}")
            elif ups and len(ups) > 1:
                st.caption(f"✅ {len(ups)} files loaded")
            else:
                st.caption("✅ loaded")
    st.session_state["reference_samples"] = samples

    st.subheader("🔑 Marker-only materials (optional)")
    st.caption(
        "Answer key and/or marking notes — visible to the model, **never shown "
        "to students**. The model verifies numerical work against the key "
        "(errors penalised once at their source), follows any marking "
        "instructions (e.g. concepts not to name in feedback), and will not "
        "quote solution values or internal targets in its notes."
    )
    marker_files = st.file_uploader(
        "Answer key / marking notes (you can add several files)",
        type=["pdf", "docx", "xlsx", "xlsb", "xls", "txt", "md", "csv"],
        key=_up_key("marker_up"),
        accept_multiple_files=True,
    )
    _read_uploads_once(marker_files, "marker_text", "_marker_file_id")
    if st.session_state["marker_text"].strip():
        st.caption(
            f"✅ Loaded ~{len(st.session_state['marker_text']):,} characters "
            f"(≈ {len(st.session_state['marker_text']) // 4:,} tokens, cached "
            "across the whole batch on Claude)."
        )
        with st.expander("Preview marker materials (as the model will see them)"):
            st.text(st.session_state["marker_text"][:5000])

    if st.button("🔍 Structure rubric into components", type="primary"):
        if not st.session_state["rubric_text"].strip():
            st.error("Upload or paste a rubric first.")
        else:
            with st.spinner("Parsing rubric…"):
                try:
                    rubric, used = structure_rubric_with_fallback(
                        _provider_chain(), st.session_state["rubric_text"]
                    )
                    st.session_state["components"] = rubric.components
                    st.caption(f"Structured with: {used}")
                except Exception as exc:
                    st.error(f"Rubric structuring failed: {exc}")

    if st.session_state["components"]:
        st.success("Rubric components (weights sum to 100):")
        st.table(
            [
                {"Component": c.name, "Weight": f"{c.weight:g}", "Assesses": c.description}
                for c in st.session_state["components"]
            ]
        )
        _sample_compression_ui()


def _sample_compression_ui() -> None:
    """Compress full reference samples into compact anchor profiles.

    The model reads each sample in full once, then grading uses the short
    profile — so small-context models (Ollama) keep the calibration signal of
    the whole sample without sending the whole (truncated) text every request.
    """
    samples = st.session_state["reference_samples"]
    has_samples = any((samples.get(k) or "").strip() for k in SAMPLE_LEVELS)
    if not has_samples:
        return

    st.markdown("**🧭 Reference samples → compact anchors** "
                "(recommended for small-context models like Ollama)")
    st.caption(
        "The model reads each FULL sample once and distils it into a ~150-word "
        "profile of the whole paper. Grading then uses the profile, so requests "
        "stay tiny without truncating the samples mid-way. Big-context models "
        "(Claude/Gemini) can skip this and read full samples directly."
    )
    if st.button("Read & compress samples"):
        progress = st.progress(0.0, text="Starting…")

        def cb(done: int, total: int, label: str) -> None:
            progress.progress(min(1.0, done / max(1, total)), text=label)

        try:
            profiles, used = summarize_samples_with_fallback(
                _provider_chain(), samples, st.session_state["components"], progress_callback=cb
            )
            st.session_state["sample_profiles"] = profiles
            # Fingerprint the sample each profile was distilled from, so a
            # slot whose file is later replaced/added falls back to its FULL
            # text instead of silently grading against a stale profile.
            st.session_state["profile_hashes"] = {
                k: _sample_fingerprint(samples.get(k, "")) for k in profiles
            }
            st.session_state["use_profiles"] = True
            st.success(f"Compressed {len(profiles)} sample(s) with {used}.")
        except Exception as exc:
            st.error(f"Sample compression failed: {exc}")
        finally:
            progress.empty()

    if st.session_state.get("sample_profiles"):
        st.session_state.setdefault("use_profiles", True)
        st.checkbox("Use compact anchor profiles for grading", key="use_profiles")
        if st.session_state["use_profiles"]:
            hashes = st.session_state.get("profile_hashes", {})
            profiles = st.session_state["sample_profiles"]
            changed = [
                SAMPLE_LEVELS[k]
                for k in SAMPLE_LEVELS
                if (samples.get(k) or "").strip()
                and (k not in profiles or hashes.get(k) != _sample_fingerprint(samples[k]))
            ]
            if changed:
                st.info(
                    "These samples were added or changed AFTER compression, so "
                    "they are sent in full until you re-run **Read & compress**: "
                    + ", ".join(changed) + "."
                )
        with st.expander("Review anchor profiles"):
            for key, label in SAMPLE_LEVELS.items():
                prof = st.session_state["sample_profiles"].get(key)
                if prof:
                    st.markdown(f"**{label}**")
                    st.write(prof)


def batch_ui() -> None:
    st.header("2 · Batch Upload — Student Reports")
    uploads = st.file_uploader(
        "Drop all reports (PDF/DOCX/XLSX/TXT) or a single .zip — a student's "
        "Word report and Excel workbook can be uploaded together",
        type=["pdf", "docx", "xlsx", "xlsb", "xls", "txt", "md", "csv", "png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True,
        key=_up_key("batch_up"),
    )
    st.session_state.setdefault("group_by_student", True)
    group_enabled = st.checkbox(
        "Group files by student (Word + Excel of the same student become ONE submission)",
        key="group_by_student",
        help="Files are grouped by the student ID in the filename (e.g. "
             "s4180007); files without an ID pair up when their names match "
             "apart from the extension. Verify the grouping below before "
             "grading.",
    )
    # Parse once per upload set; removing EVERY file clears the cohort too —
    # otherwise deleted reports would silently stay queued for grading.
    if uploads:
        ids = tuple(u.file_id for u in uploads)
        if st.session_state.get("_batch_ids") != ids:
            raw_docs: list[ParsedDocument] = []
            for up in uploads:
                raw_docs.extend(parse_upload(up))
            st.session_state["_batch_raw"] = raw_docs
            st.session_state["_batch_ids"] = ids
    elif st.session_state.pop("_batch_ids", None) is not None:
        st.session_state["_batch_raw"] = []
    raw_docs = st.session_state.get("_batch_raw", [])
    # Grouping is recomputed from the parsed files each run, so toggling the
    # checkbox re-groups instantly without re-parsing.
    st.session_state["documents"] = group_by_student(raw_docs, enabled=group_enabled)
    st.session_state["_raw_file_count"] = len(raw_docs)
    docs = st.session_state["documents"]
    if docs:
        raw_n = st.session_state.get("_raw_file_count", len(docs))
        if raw_n != len(docs):
            st.info(f"{len(docs)} submission(s) from {raw_n} files.")
        else:
            st.info(f"{len(docs)} submission(s) ready.")
        problems = [(d, text_health(d.text)) for d in docs]
        unreadable = [(d, h) for d, h in problems if h]
        if unreadable:
            def _line(d, h):
                line = f"- **{d.name}** — {HEALTH_HINTS[h]}"
                reason = read_failure_reason(d.text)
                if reason:
                    line += f"\n  - ⚙️ reason: `{reason}`"
                return line

            lines = "\n".join(_line(d, h) for d, h in unreadable)
            st.warning(
                f"⚠️ **{len(unreadable)} submission(s) have little or no readable "
                "text** and would be graded on almost nothing:\n\n" + lines +
                "\n\nFix the files (OCR / re-export / convert) and re-upload, or "
                "remove them and grade those students manually."
            )
        with st.expander(
            "Review submissions / grouping",
            expanded=(raw_n != len(docs)) or bool(unreadable),
        ):
            for d, health in problems:
                n_files = d.text.count("=== File: ") or 1
                flag = f"  ⚠️ *{HEALTH_HINTS[health]}*" if health else ""
                st.write(f"• **{d.name}** — {n_files} file(s), ~{len(d.text):,} chars{flag}")


def _sample_fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").strip().encode("utf-8", "replace")).hexdigest()


def _effective_samples() -> dict:
    """The reference samples grading will actually use.

    Always anchored to the samples currently LOADED in the slots — a deleted
    sample never keeps grading via its leftover profile, and a sample added
    after compression is never silently dropped. When compact profiles are
    enabled, a slot's profile substitutes for its full text only while the
    underlying sample is unchanged (checked via the fingerprint stored at
    compression time); otherwise the full text is sent.
    """
    samples = {
        k: v for k, v in st.session_state["reference_samples"].items()
        if (v or "").strip()
    }
    if not (st.session_state.get("use_profiles") and st.session_state.get("sample_profiles")):
        return samples
    profiles = st.session_state["sample_profiles"]
    hashes = st.session_state.get("profile_hashes", {})
    return {
        k: profiles[k]
        if k in profiles and hashes.get(k) == _sample_fingerprint(v)
        else v
        for k, v in samples.items()
    }


def _session_from_state() -> GradingSession:
    return GradingSession(
        task_text=st.session_state["task_text"],
        rubric_text=st.session_state["rubric_text"],
        components=st.session_state["components"],
        reference_samples=_effective_samples(),
        name=st.session_state["session_name"] or "Untitled session",
        marker_materials=st.session_state.get("marker_text", ""),
    )


def _estimate_ui(session: GradingSession) -> None:
    docs = st.session_state["documents"]
    provider_key = st.session_state.get("provider_key", "anthropic")
    preset = PROVIDER_PRESETS[provider_key]
    est = estimate_batch(
        provider_key=provider_key,
        provider_label=preset.label,
        model=st.session_state.get("model") or preset.default_model,
        static_context_text=static_context_text(session),
        report_texts=[d.text for d in docs],
        n_components=len(session.components),
        pricing=PROVIDER_PRICING.get(provider_key),
    )
    with st.expander("📊 Estimate cost & time (before you run)", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Reports", est.n_reports)
        cols[1].metric("Est. cost", est.cost_label, help=f"{preset.label} · {est.model}")
        cols[2].metric("Est. time", est.time_label)
        cols[3].metric(
            "Tokens in/out",
            f"{est.input_tokens_billed:,} / {est.output_tokens:,}",
        )

        # Transparency: tell the professor exactly what (if anything) is trimmed.
        cap_r, cap_s = config.MAX_REPORT_CHARS, config.MAX_SAMPLE_CHARS
        over_reports = [d.name for d in docs if cap_r and len(d.text) > cap_r]
        over_samples = [
            k for k, v in session.reference_samples.items() if v and cap_s and len(v) > cap_s
        ]
        if over_reports or over_samples:
            bits = []
            if over_reports:
                bits.append(f"{len(over_reports)} report(s)")
            if over_samples:
                bits.append(f"sample(s) {', '.join(over_samples)}")
            st.warning(
                "✂️ At the current length limit, " + " and ".join(bits)
                + " will be shortened (the model won't see the trimmed tail). "
                "Raise **📏 Document length** in the sidebar (set 0 = read "
                "everything) to grade them in full."
            )
        else:
            st.caption("✅ No truncation — every document fits; the model reads each in full.")

        st.caption(
            f"One request ≈ {est.tokens_per_request_avg:,} tokens "
            f"(largest ≈ {est.tokens_per_request_max:,}). For **Ollama**, set "
            "`OLLAMA_CONTEXT_LENGTH` above the larger number — or lower "
            "**reference sample length** in the sidebar 📏 until it fits, while "
            "keeping the report length high."
        )
        if provider_key == "ollama":
            st.warning(
                "⏱️ Local speed depends heavily on your CPU/GPU. On a 16GB "
                "laptop with no GPU, an 8B model can take **several minutes per "
                "report** (the first one also loads the model). To speed up: use "
                "a smaller model (`qwen2.5:3b`), check `ollama ps` shows GPU use, "
                "or switch to **Gemini free** for a much faster run."
            )
        if est.supports_cache:
            st.caption(
                f"Prompt caching applied: the shared context (~{est.static_tokens:,} "
                "tokens) is written once, then read at ~0.1× for every later report."
            )
        st.markdown("**Compare providers** — default model, cheapest first:")
        comparison = compare_providers(session, docs, PROVIDER_PRESETS)
        st.dataframe(
            [
                {
                    "Provider": e.provider_label,
                    "Model": e.model,
                    "Est. cost": e.cost_label,
                    "Est. time": e.time_label,
                }
                for e in comparison
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "⚠️ Indicative only: token counts are estimated locally (~4 chars/token) "
            "and the prices in `src/config.py` are approximate and change over time."
        )


def grade_ui() -> None:
    st.header("3 · Grade & Calibrate")
    ready = bool(
        st.session_state["task_text"].strip()
        and st.session_state["components"]
        and st.session_state["documents"]
    )
    if not ready:
        st.caption("Need: a task, structured rubric components, and at least one report. "
                   "Reference samples are optional.")
    else:
        anchors = [
            label for key, label in REFERENCE_KEYS
            if (_effective_samples().get(key) or "").strip()
        ]
        if not anchors:
            st.caption(
                "🎯 No reference samples — the model will grade against the "
                "rubric and task alone (fine, but anchors sharpen calibration)."
            )
        elif len(anchors) < len(REFERENCE_KEYS):
            st.caption(
                f"🎯 Calibration anchors in use: {', '.join(anchors)}. Missing "
                "levels are interpolated from these and the rubric."
            )
        _estimate_ui(_session_from_state())

    if st.button("🚀 Run Two-Pass Grading", type="primary", disabled=not ready):
        session = _session_from_state()
        progress = st.progress(0.0, text="Starting…")

        def cb(done: int, total: int, label: str) -> None:
            progress.progress(min(1.0, done / max(1, total)), text=label)

        try:
            result = grade_batch_with_fallback(
                _provider_chain(), session, st.session_state["documents"],
                calibration_config=_calibration_config(), progress_callback=cb,
            )
            st.session_state["batch_result"] = result
        except Exception as exc:
            st.error(f"Grading failed: {exc}")
        finally:
            progress.empty()

    _render_results()


def _render_results() -> None:
    result = st.session_state.get("batch_result")
    if not result:
        return

    if result.provider_label:
        if result.fell_back_from:
            st.warning(
                f"✅ Whole cohort graded by **{result.provider_label}** "
                f"(fell back from: {', '.join(result.fell_back_from)}). "
                "All reports share one consistent scale."
            )
        else:
            st.caption(f"Graded by: {result.provider_label}")
    if result.notes:
        with st.expander("Fallback / audit trail", expanded=bool(result.fell_back_from)):
            for note in result.notes:
                st.write("•", note)

    if result.errors:
        with st.expander(f"⚠️ {len(result.errors)} report(s) failed", expanded=False):
            for err in result.errors:
                st.write(err)

    # Re-calibrate: apply new mean/cap/boundary settings for free (no LLM calls).
    if result.raw_grades:
        if st.button(
            "🔄 Re-calibrate with current settings (no tokens)",
            help="Re-runs only the calibration on the already-graded cohort using "
                 "the sidebar Calibration targets. Free — the LLM is not called.",
        ):
            st.session_state["batch_result"] = recalibrate(result, _calibration_config())
            st.rerun()

    calib = result.calibration
    if calib:
        cols = st.columns(5)
        cols[0].metric("Calibrated mean", f"{calib.final_mean:.1f}%")
        cols[1].metric("HD (≥80%)", f"{calib.hd_count}", help=f"cap {calib.hd_cap}")
        cols[2].metric("DI (70-79%)", f"{calib.di_count}", help=f"cap {calib.di_cap}")
        cols[3].metric("Fails (<50%)", f"{calib.fail_count}")
        cols[4].metric("Reports", f"{calib.n}")
        if calib.low_discrimination:
            st.warning(
                f"⚠️ **Low differentiation.** The model gave very similar raw "
                f"scores (raw mean {calib.raw_mean:.1f}%, spread ±{calib.raw_std:.1f}; "
                f"only {calib.distinct_finals} distinct final grades for {calib.n} "
                "reports). It isn't really separating students — calibration can "
                "only rank what the model produced. **Re-grade with a stronger "
                "model** (e.g. Claude, or a paid model like gpt-4o-mini/DeepSeek) "
                "for grades that discriminate."
            )

    if result.reports:
        st.subheader("Calibrated grades")
        st.dataframe(
            [
                {
                    "Student": r.student_identifier,
                    "Raw": f"{r.raw_total:g}",
                    "Final": f"{r.final_total:g}",
                    "Band": r.band,
                }
                for r in result.reports
            ],
            use_container_width=True,
        )

        docx_bytes = build_grade_document(
            result.reports,
            st.session_state["session_name"] or "Untitled session",
            calib,
            graded_by=result.provider_label,
        )
        st.download_button(
            "📄 Download Word grade report (.docx)",
            data=docx_bytes,
            file_name=f"grades_{(st.session_state['session_name'] or 'session').replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )

        for r in result.reports:
            with st.expander(f"{r.student_identifier} - {r.final_total:g}/100 ({r.band})"):
                for c in r.components:
                    st.markdown(f"**{sanitize_text(c.name)}** - {c.score:g}/{c.max_score:g}")
                    for note in c.notes:
                        st.text(sanitize_text(note))


def main() -> None:
    _init_state()
    st.title("🎓 AI Grading Studio")
    st.caption(
        "Course-agnostic, rubric-aligned grading with a Two-Pass statistical "
        "calibration engine."
    )
    sidebar()
    calibration_data_ui()
    st.divider()
    batch_ui()
    st.divider()
    grade_ui()


if __name__ == "__main__":
    main()
