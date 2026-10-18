"""Batch grouping warning for folders that may hold several students, and
Excel formulas saved without a computed result."""
from __future__ import annotations

import io
import pathlib
import re
import zipfile

import pytest

from src.document_parser import ParsedDocument, auto_group_labels, parse_bytes
from src.providers import PROVIDER_PRESETS

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    # app.py loads the real .env on import; set first, load_dotenv won't override.
    for preset in PROVIDER_PRESETS.values():
        if preset.api_key_env:
            monkeypatch.setenv(preset.api_key_env, "dummy-not-a-real-key")


# --- Grouping: an ID-less folder of different names may be several students --

LATE = [
    "Late submissions/Alice Nguyen.docx",
    "Late submissions/Bob Tran.docx",
    "Late submissions/Carol Le.PDF",
]


def _labels(names, enabled=True):
    return auto_group_labels([ParsedDocument(n, "text") for n in names], enabled)


def test_folder_of_differently_named_reports_is_flagged():
    import app

    names = LATE + [
        "Dan_123_assignsubmission_file_/report.docx",   # one report + workbook
        "Dan_123_assignsubmission_file_/report.xlsx",
        "Fay/Essay.docx", "Fay/essay (1).pdf",           # one name, two formats
    ]
    assert app._mixed_name_groups(_labels(names), {}) == {"Alice Nguyen": LATE}


def test_no_flag_when_grouping_is_off_or_files_were_relabelled():
    import app

    assert app._mixed_name_groups(_labels(LATE, enabled=False), {}) == {}
    split = {LATE[1]: "Bob Tran", LATE[2]: "Carol Le"}
    assert app._mixed_name_groups(_labels(LATE), split) == {}
    # Splitting off one student still leaves two names in the group.
    assert app._mixed_name_groups(_labels(LATE), {LATE[2]: "Carol Le"}) == {
        "Alice Nguyen": LATE[:2]
    }


def _docx(text: str) -> bytes:
    import docx

    document = docx.Document()
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_batch_ui_warns_about_a_zipped_folder_of_students():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in ("Alice Nguyen", "Bob Tran", "Carol Le"):
            zf.writestr(f"Late submissions/{name}.docx", _docx(f"Report by {name}. " * 30))
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    uploader = [u for u in at.file_uploader if "Drop all reports" in u.label][0]
    uploader.set_value([("Late submissions.zip", buf.getvalue(), "application/zip")])
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    expected = (
        "'Alice Nguyen' groups 3 Word/PDF files with different names "
        "(Alice Nguyen.docx, Bob Tran.docx, Carol Le.docx)"
    )
    assert expected in " | ".join(w.value for w in at.warning)

    [c for c in at.checkbox if c.key == "group_by_student"][0].uncheck()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Word/PDF files with different names" not in " | ".join(
        w.value for w in at.warning
    )


# --- Excel: formulas without a saved result show the formula text ----------

def _workbook(cached: dict) -> bytes:
    """An openpyxl workbook (formulas saved with no result), with the results
    in ``cached`` (cell -> value) written in afterwards as Excel would save
    them; "" is a text result, saved as t="str" with an empty value."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Rate", 0.1])
    ws.append(["Total", "=SUM(C1:C2)"])
    ws.append(["NPV", "=NPV(B1,C3:C7)"])
    ws.append(["Note", '=IF(B1>1,"high","")'])
    raw = io.BytesIO()
    wb.save(raw)
    out = io.BytesIO()
    with zipfile.ZipFile(raw) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "xl/worksheets/sheet1.xml":
                for ref, value in cached.items():
                    kind = ' t="str"' if value == "" else ""
                    data, n = re.subn(
                        rf'<c r="{ref}">(<f>[^<]*</f>)<v></v>'.encode(),
                        rf'<c r="{ref}"{kind}>\g<1><v>{value}</v>'.encode(),
                        data,
                    )
                    assert n == 1
            dst.writestr(info, data)
    return out.getvalue()


def test_formula_without_saved_result_reads_as_its_formula():
    text = parse_bytes("work.xlsx", _workbook({"B2": 42}))[0].text
    assert text.splitlines() == [
        "Rate | 0.1", "Total | 42", "NPV | =NPV(B1,C3:C7)", 'Note | =IF(B1>1,"high","")',
    ]


def test_workbook_with_all_results_saved_is_loaded_once(monkeypatch):
    import openpyxl

    calls = []
    real = openpyxl.load_workbook
    monkeypatch.setattr(
        openpyxl, "load_workbook", lambda *a, **k: calls.append(k) or real(*a, **k)
    )
    data = _workbook({"B2": 42, "B3": 7.5, "B4": ""})
    text = parse_bytes("work.xlsm", data)[0].text
    assert text.splitlines() == ["Rate | 0.1", "Total | 42", "NPV | 7.5", "Note"]
    assert len(calls) == 1
