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

from src.config import DEFAULT_CALIBRATION, DEFAULT_EFFORT
from src.docx_export import build_grade_document
from src.document_parser import ParsedDocument, parse_upload
from src.grading_engine import GradingSession, grade_batch
from src.providers import PROVIDER_PRESETS, build_provider

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


def _grader():
    """Build the grading back-end from the sidebar provider selection."""
    return build_provider(
        st.session_state.get("provider_key", "anthropic"),
        model=st.session_state.get("model") or None,
        api_key=st.session_state.get("api_key") or None,
        base_url=st.session_state.get("base_url") if st.session_state.get("base_url") else None,
        effort=st.session_state.get("effort", DEFAULT_EFFORT),
        temperature=st.session_state.get("temperature", 0.2),
    )


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
        if task_file:
            st.session_state["task_text"] = _first_text(task_file)
        st.session_state["task_text"] = st.text_area(
            "Task text", value=st.session_state["task_text"], height=180
        )

    with col_rubric:
        st.subheader("📐 The Rubric")
        rubric_file = st.file_uploader(
            "Grading criteria & weightings", type=["pdf", "docx", "txt", "md"], key="rubric_up"
        )
        if rubric_file:
            st.session_state["rubric_text"] = _first_text(rubric_file)
        st.session_state["rubric_text"] = st.text_area(
            "Rubric text", value=st.session_state["rubric_text"], height=180
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
                    rubric = _grader().structure_rubric(st.session_state["rubric_text"])
                    st.session_state["components"] = rubric.components
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


def grade_ui() -> None:
    st.header("3 · Grade & Calibrate")
    ready = bool(
        st.session_state["task_text"].strip()
        and st.session_state["components"]
        and st.session_state["documents"]
    )
    if not ready:
        st.caption("Need: a task, structured rubric components, and at least one report.")

    if st.button("🚀 Run Two-Pass Grading", type="primary", disabled=not ready):
        session = GradingSession(
            task_text=st.session_state["task_text"],
            rubric_text=st.session_state["rubric_text"],
            components=st.session_state["components"],
            reference_samples=st.session_state["reference_samples"],
            name=st.session_state["session_name"] or "Untitled session",
        )
        progress = st.progress(0.0, text="Starting…")

        def cb(done: int, total: int, label: str) -> None:
            progress.progress(min(1.0, done / max(1, total)), text=label)

        try:
            result = grade_batch(_grader(), session, st.session_state["documents"], progress_callback=cb)
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
