"""Streamlit UI for the AI grading application.

Workflow:
  1. New Grading Session — name it.
  2. Calibration data — upload Task, Rubric, and four reference samples.
     Structuring the rubric turns it into weighted components.
  3. Batch upload — drop in all student reports (or a .zip).
  4. Grade — runs the Two-Pass engine and shows the calibrated cohort.
  5. Download — a structured .docx grade report.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src import config
from src.config import DEFAULT_CALIBRATION, DEFAULT_EFFORT, PROVIDER_PRICING

# Baseline caps captured before any runtime override (used as UI defaults).
_BASE_REPORT_CAP = config.MAX_REPORT_CHARS
_BASE_SAMPLE_CAP = config.MAX_SAMPLE_CHARS
from src.docx_export import build_grade_document
from src.document_parser import ParsedDocument, parse_upload
from src.estimator import compare_providers, estimate_batch, static_context_text
from src.grading_engine import (
    GradingSession,
    ProviderSpec,
    grade_batch_with_fallback,
    structure_rubric_with_fallback,
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


def _first_text(uploaded) -> str:
    docs = parse_upload(uploaded)
    return docs[0].text if docs else ""


def _read_uploaded_once(uploaded, text_key: str, file_id_key: str) -> None:
    """Read an uploaded file into session state exactly once per upload.

    Reading only when the file id changes means a later manual paste into the
    text area is not clobbered on every Streamlit rerun.
    """
    if uploaded is None:
        return
    if st.session_state.get(file_id_key) != uploaded.file_id:
        st.session_state[text_key] = _first_text(uploaded)
        st.session_state[file_id_key] = uploaded.file_id


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

def sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Provider")

        keys = list(PROVIDER_PRESETS.keys())

        def _fmt(k: str) -> str:
            p = PROVIDER_PRESETS[k]
            tag = " · 🆓" if p.free else ""
            return f"{p.label}{tag}"

        provider_key = st.selectbox(
            "Grading engine", keys, format_func=_fmt,
            index=keys.index(st.session_state.get("provider_key", "anthropic")),
            help="Anthropic is paid. Gemini & Groq have free tiers; Ollama is "
                 "fully free and runs locally.",
        )
        st.session_state["provider_key"] = provider_key
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
        st.session_state["model"] = st.text_input("Model", value=st.session_state.get("model", ""))

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
            st.session_state["api_key"] = st.text_input(
                label, type="password", value=st.session_state.get("api_key", ""),
                help="Or set it in your environment / .env file.",
            )
            if preset.api_key_env and not st.session_state["api_key"]:
                st.warning(f"Set {preset.api_key_env} or paste a key above.")

        # Base URL — only for custom endpoints.
        if provider_key == "custom":
            st.session_state["base_url"] = st.text_input(
                "Base URL", value=st.session_state.get("base_url", ""),
                placeholder="https://your-endpoint/v1",
            )
        else:
            st.session_state["base_url"] = ""

        # Generation controls differ by kind.
        if preset.kind == "anthropic":
            st.session_state["effort"] = st.select_slider(
                "Effort", options=["low", "medium", "high", "max"],
                value=st.session_state.get("effort", DEFAULT_EFFORT),
            )
        else:
            st.session_state["temperature"] = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.2), 0.05,
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
        st.session_state["fallback_keys"] = st.multiselect(
            "Fallback providers (in order)",
            options=usable,
            default=[k for k in st.session_state.get("fallback_keys", []) if k in usable],
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
        report_cap = st.number_input(
            "Student report — max characters (0 = full)",
            min_value=0, max_value=400_000,
            value=int(st.session_state.get("report_cap", _BASE_REPORT_CAP)), step=2000,
            help="The work being graded — keep this HIGH so the model reads the "
                 "whole report. A 2,500-word report is ~16,000 chars.",
        )
        sample_cap = st.number_input(
            "Each reference sample — max characters (0 = full)",
            min_value=0, max_value=400_000,
            value=int(st.session_state.get("sample_cap", _BASE_SAMPLE_CAP)), step=2000,
            help="Samples are calibration anchors — an excerpt is enough. LOWER "
                 "this first to shrink each request (biggest lever for small "
                 "contexts like Ollama). 4 samples are sent every request.",
        )
        st.session_state["report_cap"] = report_cap
        st.session_state["sample_cap"] = sample_cap
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
        c = DEFAULT_CALIBRATION
        st.caption(
            f"Mean {c.target_mean_low:g}–{c.target_mean_high:g}% · "
            f"HD ≤{c.hd_cap_fraction:.0%} · DI ≤{c.di_cap_fraction:.0%} · "
            f"fails minimised (<{c.fail_threshold:g}%)"
        )

        st.divider()
        if st.button("🆕 New Grading Session", use_container_width=True):
            for key in (
                "session_name", "task_text", "rubric_text",
                "reference_samples", "components", "documents", "batch_result",
            ):
                st.session_state.pop(key, None)
            _init_state()
            st.rerun()


def calibration_data_ui() -> None:
    st.header("1 · Session & Calibration Data")
    st.session_state["session_name"] = st.text_input(
        "Session / assignment name",
        value=st.session_state["session_name"],
        placeholder="e.g. ECON301 — Market Analysis Essay (S1 2026)",
    )

    col_task, col_rubric = st.columns(2)
    with col_task:
        st.subheader("📋 The Task")
        task_file = st.file_uploader(
            "Assignment requirements", type=["pdf", "docx", "txt", "md"], key="task_up"
        )
        _read_uploaded_once(task_file, "task_text", "_task_file_id")
        st.session_state["task_text"] = st.text_area(
            "Task text (you can also paste/edit here)",
            value=st.session_state["task_text"], height=180,
        )

    with col_rubric:
        st.subheader("📐 The Rubric")
        rubric_file = st.file_uploader(
            "Grading criteria & weightings", type=["pdf", "docx", "txt", "md"], key="rubric_up"
        )
        _read_uploaded_once(rubric_file, "rubric_text", "_rubric_file_id")
        if rubric_file is not None and not st.session_state["rubric_text"].strip():
            st.warning(
                "Couldn't extract text from this file (it may be scanned or laid "
                "out in a text box). Paste the rubric text below instead."
            )
        st.session_state["rubric_text"] = st.text_area(
            "Rubric text (you can also paste/edit here)",
            value=st.session_state["rubric_text"], height=180,
        )

    st.subheader("🎯 Reference Samples (calibration anchors)")
    cols = st.columns(4)
    samples = st.session_state["reference_samples"]
    for (key, label), col in zip(REFERENCE_KEYS, cols):
        with col:
            up = st.file_uploader(label, type=["pdf", "docx", "txt", "md"], key=f"ref_{key}")
            if up:
                samples[key] = _first_text(up)
            status = "✅ loaded" if samples.get(key) else "— empty"
            st.caption(status)
    st.session_state["reference_samples"] = samples

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


def batch_ui() -> None:
    st.header("2 · Batch Upload — Student Reports")
    uploads = st.file_uploader(
        "Drop all reports (PDF/DOCX/TXT) or a single .zip",
        type=["pdf", "docx", "txt", "md", "zip"],
        accept_multiple_files=True,
        key="batch_up",
    )
    if uploads:
        docs: list[ParsedDocument] = []
        for up in uploads:
            docs.extend(parse_upload(up))
        st.session_state["documents"] = docs
    docs = st.session_state["documents"]
    if docs:
        st.info(f"{len(docs)} report(s) ready: " + ", ".join(d.name for d in docs[:8])
                + (" …" if len(docs) > 8 else ""))


def _session_from_state() -> GradingSession:
    return GradingSession(
        task_text=st.session_state["task_text"],
        rubric_text=st.session_state["rubric_text"],
        components=st.session_state["components"],
        reference_samples=st.session_state["reference_samples"],
        name=st.session_state["session_name"] or "Untitled session",
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
        st.caption("Need: a task, structured rubric components, and at least one report.")
    else:
        _estimate_ui(_session_from_state())

    if st.button("🚀 Run Two-Pass Grading", type="primary", disabled=not ready):
        session = _session_from_state()
        progress = st.progress(0.0, text="Starting…")

        def cb(done: int, total: int, label: str) -> None:
            progress.progress(min(1.0, done / max(1, total)), text=label)

        try:
            result = grade_batch_with_fallback(
                _provider_chain(), session, st.session_state["documents"], progress_callback=cb
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

    calib = result.calibration
    if calib:
        cols = st.columns(5)
        cols[0].metric("Calibrated mean", f"{calib.final_mean:.1f}%")
        cols[1].metric("HD (≥80%)", f"{calib.hd_count}", help=f"cap {calib.hd_cap}")
        cols[2].metric("DI (70-79%)", f"{calib.di_count}", help=f"cap {calib.di_cap}")
        cols[3].metric("Fails (<50%)", f"{calib.fail_count}")
        cols[4].metric("Reports", f"{calib.n}")

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
            with st.expander(f"{r.student_identifier} — {r.final_total:g}/100 ({r.band})"):
                for c in r.components:
                    st.markdown(f"**{c.name}** — {c.score:g}/{c.max_score:g}")
                    for note in c.notes:
                        st.markdown(f"- {note}")


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
