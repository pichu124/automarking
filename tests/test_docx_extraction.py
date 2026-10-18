"""Word body text must reach the grader complete, once, and in reading order.

Fixtures are built with python-docx and then given hand-written XML for the
constructs python-docx cannot author (content controls, text boxes,
equations, tracked changes, footnotes).
"""
from __future__ import annotations

import io
import zipfile

import docx
import pytest
from docx.oxml import parse_xml

from src.document_parser import parse_bytes

_NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
)


def _p(text: str) -> str:
    return f"<w:p><w:r><w:t xml:space='preserve'>{text}</w:t></w:r></w:p>"


def _docx(body_xml: str, header: str = "") -> bytes:
    """A real python-docx package whose body holds `body_xml`."""
    d = docx.Document()
    if header:
        d.sections[0].header.paragraphs[0].text = header
    body = d.element.body
    for child in list(parse_xml(f"<w:body {_NS}>{body_xml}</w:body>")):
        body.insert(len(body) - 1, child)  # keep w:sectPr last
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _rewrite(data: bytes, edits: dict, extra: dict | None = None) -> bytes:
    """Copy a docx zip, appending XML before closing tags and adding parts.

    `edits` maps member name -> (closing tag, text inserted before it).
    """
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            blob = zin.read(item.filename)
            if item.filename in edits:
                close, insert = edits[item.filename]
                blob = blob.replace(close.encode(), (insert + close).encode())
            zout.writestr(item, blob)
        for name, xml in (extra or {}).items():
            zout.writestr(name, xml)
    return out.getvalue()


def _with_notes(data: bytes, footnotes: str, endnotes: str) -> bytes:
    types, rels = "", ""
    for kind in ("footnotes", "endnotes"):
        types += (
            f'<Override PartName="/word/{kind}.xml" ContentType="application/'
            f'vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml"/>'
        )
        rels += (
            f'<Relationship Id="rId_{kind}" Type="http://schemas.openxmlformats'
            f'.org/officeDocument/2006/relationships/{kind}" Target="{kind}.xml"/>'
        )
    return _rewrite(
        data,
        {
            "[Content_Types].xml": ("</Types>", types),
            "word/_rels/document.xml.rels": ("</Relationships>", rels),
        },
        {
            "word/footnotes.xml": f"<w:footnotes {_NS}>{footnotes}</w:footnotes>",
            "word/endnotes.xml": f"<w:endnotes {_NS}>{endnotes}</w:endnotes>",
        },
    )


def _with_dangling_rel(data: bytes) -> bytes:
    """python-docx refuses this package; only the raw-XML reader can open it."""
    ghost = (
        '<Relationship Id="rIdGhost" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/image" Target="media/image9.png"/>'
    )
    return _rewrite(data, {"word/_rels/document.xml.rels": ("</Relationships>", ghost)})


def _text(data: bytes) -> str:
    return parse_bytes("report.docx", data)[0].text


_TEXT_BOX = (
    "<w:p><w:r><w:t>Figure caption</w:t></w:r><w:r><mc:AlternateContent>"
    "<mc:Choice Requires='wps'><w:drawing><wps:txbx><w:txbxContent>"
    + _p("Text box answer")
    + "</w:txbxContent></wps:txbx></w:drawing></mc:Choice>"
    "<mc:Fallback><w:pict><v:shape><v:textbox><w:txbxContent>"
    + _p("Text box answer")
    + "</w:txbxContent></v:textbox></v:shape></w:pict></mc:Fallback>"
    "</mc:AlternateContent></w:r></w:p>"
)

_RICH_BODY = (
    _p("Introduction paragraph.")
    # Block-level content control (a template answer box).
    + "<w:sdt><w:sdtPr><w:alias w:val='Answer 1'/></w:sdtPr><w:sdtContent>"
    + _p("Answer typed inside the content control.")
    + "</w:sdtContent></w:sdt>"
    # Inline content control inside an ordinary paragraph.
    + "<w:p><w:r><w:t xml:space='preserve'>Student name: </w:t></w:r>"
    "<w:sdt><w:sdtContent><w:r><w:t>Alice Nguyen</w:t></w:r></w:sdtContent></w:sdt></w:p>"
    + _TEXT_BOX
    # Equation: "NPV = " followed by an m:oMath object.
    + "<w:p><w:r><w:t xml:space='preserve'>NPV = </w:t></w:r>"
    "<m:oMath><m:r><m:t>CF/(1+r)</m:t></m:r></m:oMath></w:p>"
    # Tracked changes and a field: deleted text and the field code are not
    # part of what the student's document says; the field result is.
    + "<w:p><w:r><w:t xml:space='preserve'>Rates rose </w:t></w:r>"
    "<w:del w:id='1' w:author='S'><w:r><w:delText>sharply </w:delText></w:r></w:del>"
    "<w:ins w:id='2' w:author='S'><w:r><w:t xml:space='preserve'>modestly </w:t></w:r></w:ins>"
    "<w:r><w:t xml:space='preserve'>see page </w:t></w:r>"
    "<w:r><w:fldChar w:fldCharType='begin'/></w:r>"
    "<w:r><w:instrText> PAGEREF _Ref1 </w:instrText></w:r>"
    "<w:r><w:fldChar w:fldCharType='separate'/></w:r>"
    "<w:r><w:t>4</w:t></w:r><w:r><w:fldChar w:fldCharType='end'/></w:r></w:p>"
    + "<w:tbl><w:tr><w:tc>" + _p("Criterion") + "</w:tc><w:tc>" + _p("Weight")
    + "</w:tc></w:tr><w:tr><w:tc>" + _p("Analysis") + "</w:tc><w:tc>" + _p("60")
    + "</w:tc></w:tr></w:tbl>"
    + _p("Conclusion paragraph.")
)

_FOOTNOTES = (
    "<w:footnote w:type='separator' w:id='-1'><w:p><w:r><w:separator/></w:r></w:p></w:footnote>"
    "<w:footnote w:type='continuationSeparator' w:id='0'>"
    "<w:p><w:r><w:t>SEPARATOR NOISE</w:t></w:r></w:p></w:footnote>"
    "<w:footnote w:id='1'>" + _p("Smith (2020), Corporate Finance, p. 41.") + "</w:footnote>"
)
_ENDNOTES = "<w:endnote w:id='1'>" + _p("Jones, R. (2019) Rates. Wiley.") + "</w:endnote>"


def _rich_docx() -> bytes:
    return _with_notes(_docx(_RICH_BODY, header="s4180007 FIN2001"), _FOOTNOTES, _ENDNOTES)


def test_content_controls_are_read_alongside_body_text():
    text = _text(_rich_docx())
    assert "Answer typed inside the content control." in text
    assert "Student name: Alice Nguyen" in text


def test_text_box_is_read_once():
    text = _text(_rich_docx())
    assert text.count("Text box answer") == 1
    assert text.index("Figure caption") < text.index("Text box answer")


def test_equation_text_is_kept():
    assert "NPV = CF/(1+r)" in _text(_rich_docx())


def test_tracked_deletions_and_field_codes_are_skipped():
    text = _text(_rich_docx())
    assert "Rates rose modestly see page 4" in text
    assert "sharply" not in text
    assert "PAGEREF" not in text


def test_tables_stay_in_document_order():
    text = _text(_rich_docx())
    assert "Criterion | Weight\nAnalysis | 60" in text
    assert text.index("Introduction") < text.index("Criterion | Weight")
    assert text.index("Analysis | 60") < text.index("Conclusion paragraph.")


def test_footnotes_endnotes_and_header_follow_the_body():
    text = _text(_rich_docx())
    assert "[Footnotes]\nSmith (2020), Corporate Finance, p. 41." in text
    assert "[Endnotes]\nJones, R. (2019) Rates. Wiley." in text
    assert "[Headers and footers]\ns4180007 FIN2001" in text
    assert "SEPARATOR NOISE" not in text
    assert text.index("Conclusion paragraph.") < text.index("[Footnotes]")


def test_invalid_xml_characters_do_not_lose_the_document():
    # Some generators write raw control characters, which no strict XML
    # parser (python-docx included) accepts. The text around them must survive.
    data = _rewrite(
        _docx(_p("Before MARK after.") + _p("Second paragraph.")),
        {"word/document.xml": ("MARK", "\x01")},  # -> "Before \x01MARK after."
    )
    text = _text(data)
    assert "[Could not read" not in text
    assert "Before MARK after." in text and "Second paragraph." in text


def test_broken_package_reads_exactly_like_an_intact_one():
    # A dangling relationship makes python-docx refuse the package, so the
    # raw-XML reader takes over; it must not lose or duplicate anything.
    intact = _rich_docx()
    assert _text(_with_dangling_rel(intact)) == _text(intact)


def _replace_bytes(data: bytes, pairs, members=lambda name: True) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            blob = zin.read(item.filename)
            if members(item.filename):
                for old, new in pairs:
                    blob = blob.replace(old, new)
            zout.writestr(item, blob)
    return out.getvalue()


_TO_STRICT = (
    (b"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
     b"http://purl.oclc.org/ooxml/wordprocessingml/main"),
    (b"http://schemas.openxmlformats.org/officeDocument/2006/math",
     b"http://purl.oclc.org/ooxml/officeDocument/math"),
)
_STRICT_RELS = (
    (b"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
     b"http://purl.oclc.org/ooxml/officeDocument/relationships"),
)


@pytest.mark.parametrize("strict_rels", [True, False])
def test_strict_open_xml_reads_like_transitional(strict_rels):
    # Word's "Strict Open XML Document" save option renames every namespace;
    # python-docx can't open it (or opens it without a body), so the raw-XML
    # reader must map the names back rather than return nothing.
    intact = _rich_docx()
    pairs = _TO_STRICT + (_STRICT_RELS if strict_rels else ())
    strict = _replace_bytes(intact, pairs)
    assert _text(strict) == _text(intact)


@pytest.mark.parametrize("junk", [b"", b"\x00\x01 not xml at all"])
def test_unreadable_footnotes_part_does_not_lose_the_body(junk):
    data = _with_notes(_docx(_p("Body survives.")), "", "")
    data = _replace_bytes(
        data, [(f"<w:footnotes {_NS}></w:footnotes>".encode(), junk)],
        members=lambda name: name == "word/footnotes.xml",
    )
    text = _text(data)
    assert "[Could not read" not in text
    assert "Body survives." in text
