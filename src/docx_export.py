"""Generate the downloadable Word (.docx) grade report.

The output replicates the rubric structure for every student: identifier, a
component-by-component table (score / max / professor's notes), the total out
of 100, and the band. A cohort summary page leads the document so the professor
can see the calibrated distribution at a glance.
"""
from __future__ import annotations

import io
import re
import unicodedata
from typing import List, Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .calibration import CalibrationReport
from .models import CalibratedReport


_HEADER_FILL = "2E4057"


# Typographic characters LLMs emit, mapped to plain keyboard equivalents.
_CHAR_REPLACEMENTS = {
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",  # dashes/minus
    "‘": "'", "’": "'", "‚": "'", "′": "'", "ʼ": "'",  # single quotes
    "“": '"', "”": '"', "„": '"', "″": '"',                # double quotes
    "…": "...",                                                            # ellipsis
    "≈": "~", "≅": "~", "≃": "~", "∼": "~",                 # approximately
    "≠": "!=", "≤": "<=", "≥": ">=",                             # comparisons
    "±": "+/-", "×": "x", "÷": "/",                              # math
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>",            # arrows
    "•": "-", "·": "-", "‧": "-", "●": "-",                 # bullets/dots
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",   # spaces
    "°": " deg", "™": "(TM)", "®": "(R)", "©": "(C)",       # symbols
    "€": "EUR", "£": "GBP", "¥": "JPY", "₫": "VND",         # currency
    # Letters with no decomposition (NFKD leaves them intact), notably
    # Vietnamese d-with-stroke - without these the letter is dropped entirely.
    "đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L",
}


def sanitize_text(text: str) -> str:
    """Return plain keyboard (ASCII) text.

    Replaces smart quotes, dashes, arrows, the approximately sign, etc. with
    keyboard equivalents; transliterates any accented letters to their base
    form; drops anything else non-ASCII. Keeps the text readable with only
    characters you can type.
    """
    if not text:
        return text
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Accented letters -> base ASCII (e.g. "Pham" from "Phạm").
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Anything still non-ASCII becomes a space, then collapse runs of spaces.
    text = "".join(c if ord(c) < 128 else " " for c in text)
    # Strip control characters (except tab/newline): lxml rejects them, so one
    # stray \x0b from a model or PDF would otherwise crash the export at the
    # very end of a paid grading run.
    text = "".join(c for c in text if c in "\t\n" or 32 <= ord(c) < 127)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def build_grade_document(
    reports: List[CalibratedReport],
    session_name: str,
    calibration: Optional[CalibrationReport] = None,
    graded_by: str = "",
) -> bytes:
    """Build the .docx and return its bytes (ready for download)."""
    doc = Document()
    _build_styles(doc)

    title = doc.add_heading(sanitize_text(f"Grade Report - {session_name}"), level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if calibration is not None:
        _add_cohort_summary(doc, calibration, graded_by)

    for i, report in enumerate(reports):
        if i > 0 or calibration is not None:
            doc.add_page_break()
        _add_student_section(doc, report)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _build_styles(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)


def _add_cohort_summary(doc: Document, calib: CalibrationReport, graded_by: str = "") -> None:
    doc.add_heading("Cohort Summary", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"

    def row(label: str, value: str) -> None:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value

    if graded_by:
        row("Graded by", sanitize_text(graded_by))
    row("Reports graded", str(calib.n))
    row("Calibrated mean", f"{calib.final_mean:.2f}%  (raw {calib.raw_mean:.2f}%)")
    row("High Distinction (>=80%)", f"{calib.hd_count}  (cap {calib.hd_cap})")
    widened = " - widened, no HD in cohort" if getattr(calib, "di_cap_widened", False) else ""
    row("Distinction (70-79%)", f"{calib.di_count}  (cap {calib.di_cap}{widened})")
    row("Fails (<50%)", str(calib.fail_count))
    doc.add_paragraph()

    if calib.low_discrimination:
        warn = doc.add_paragraph()
        run = warn.add_run(
            "Low differentiation: the model gave very similar raw scores "
            f"(spread +/-{calib.raw_std:.1f}), so the calibrated grades barely "
            "separate students. Consider re-grading with a stronger model."
        )
        run.italic = True
        run.font.color.rgb = RGBColor.from_string("B00020")
        doc.add_paragraph()


# Column widths for the student table (A4 portrait, ~16cm usable). Narrow
# Component/Score/Max so Professor's Notes gets most of the page.
_STUDENT_COL_WIDTHS = (Cm(3.8), Cm(1.3), Cm(1.2), Cm(9.6))


def _set_fixed_widths(table, widths) -> None:
    """Force fixed column widths (Word ignores plain column widths unless the
    table layout is fixed and every cell in the column carries the width)."""
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    for column, width in zip(table.columns, widths):
        column.width = width
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width


def _add_student_section(doc: Document, report: CalibratedReport) -> None:
    # A fully non-Latin identifier (Chinese/Arabic/emoji) sanitises to "";
    # fall back to the source filename so the section keeps its identity —
    # an empty heading has no runs and styling it would crash the export.
    name = (
        sanitize_text(report.student_identifier)
        or sanitize_text(report.source_name)
        or "(unnamed student)"
    )
    heading = doc.add_heading(name, level=1)
    if heading.runs:
        heading.runs[0].font.color.rgb = RGBColor.from_string(_HEADER_FILL)

    meta = doc.add_paragraph()
    meta.add_run("Source: ").bold = True
    meta.add_run(sanitize_text(report.source_name))

    # Component table: Component | Score | Max | Professor's Notes
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for cell, label in zip(hdr, ("Component", "Score", "Max", "Professor's Notes")):
        run = cell.paragraphs[0].add_run(label)
        run.bold = True

    for component in report.components:
        cells = table.add_row().cells
        cells[0].text = sanitize_text(component.name)
        cells[1].text = f"{component.score:g}"
        cells[2].text = f"{component.max_score:g}"
        notes_cell = cells[3]
        notes = [sanitize_text(nt) for nt in (component.notes or ["-"])]
        # Plain lines (no bullet glyph): reuse the cell's default paragraph for
        # the first note so there is no blank line at the top of the cell.
        first = notes_cell.paragraphs[0]
        first.text = notes[0]
        first.paragraph_format.space_after = Pt(0)
        for note in notes[1:]:
            p = notes_cell.add_paragraph(note)
            p.paragraph_format.space_after = Pt(0)

    # Apply after all rows exist — every cell must carry the column width.
    _set_fixed_widths(table, _STUDENT_COL_WIDTHS)

    # Total line
    total_p = doc.add_paragraph()
    total_run = total_p.add_run(
        sanitize_text(f"Total: {report.final_total:g} / 100 - {report.band}")
    )
    total_run.bold = True
    total_run.font.size = Pt(12)
    total_run.font.color.rgb = RGBColor.from_string(_HEADER_FILL)
