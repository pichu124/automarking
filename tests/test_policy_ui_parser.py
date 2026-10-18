"""Batch grouping warning for folders that may hold several students, and
Excel formulas saved without a computed result."""
from __future__ import annotations

import io
import re
import zipfile

from src.document_parser import parse_bytes


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
