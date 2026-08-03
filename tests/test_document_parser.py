"""Document parser resilience: a bad file must never crash the app."""
from __future__ import annotations

import io
import zipfile

from src.document_parser import _docx_xml_text, _read_docx, parse_bytes


def _docx_with_textbox() -> bytes:
    """A minimal docx whose only text lives in a text box (no body paragraphs)."""
    document_xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r></w:r></w:p>"  # empty body paragraph
        "<w:txbxContent>"
        "<w:p><w:r><w:t>Criterion A weight 60</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Criterion B weight 40</w:t></w:r></w:p>"
        "</w:txbxContent>"
        "</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def test_docx_xml_fallback_reads_textbox():
    text = _docx_xml_text(_docx_with_textbox())
    assert "Criterion A weight 60" in text
    assert "Criterion B weight 40" in text


def test_xlsx_rubric_reads_rows():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Rubric"
    ws.append(["Criterion", "Weight"])
    ws.append(["Analysis of market mechanisms", 60])
    ws.append(["Writing & referencing", 40])
    buf = io.BytesIO()
    wb.save(buf)

    docs = parse_bytes("rubric.xlsx", buf.getvalue())
    assert len(docs) == 1
    text = docs[0].text
    assert "Analysis of market mechanisms | 60" in text
    assert "Writing & referencing | 40" in text


def test_corrupt_xlsx_returns_placeholder_not_crash():
    docs = parse_bytes("broken.xlsx", b"this is not a real xlsx file")
    assert len(docs) == 1
    assert "Could not read" in docs[0].text


def test_corrupt_pdf_returns_placeholder_not_crash():
    docs = parse_bytes("broken.pdf", b"%PDF-1.4 this is not a real pdf body")
    assert len(docs) == 1
    assert docs[0].name == "broken.pdf"
    assert "Could not read" in docs[0].text  # graceful placeholder, no exception


def test_plain_text_still_reads():
    docs = parse_bytes("essay.txt", b"Hello, this is a report.")
    assert docs[0].text == "Hello, this is a report."


def test_garbage_pdf_bytes_do_not_raise():
    # Random bytes with a .pdf extension — must be handled, not raised.
    docs = parse_bytes("x.pdf", bytes(range(256)))
    assert len(docs) == 1
    assert isinstance(docs[0].text, str)
