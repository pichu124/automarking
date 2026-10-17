"""Parser safety: archives, PDFs and encrypted Office files from untrusted
uploads must neither exhaust the server nor silently lose a student's work."""
from __future__ import annotations

import io
import sys
import zipfile

from src import document_parser as dp
from src.document_parser import (
    auto_group_labels,
    group_by_labels,
    parse_bytes,
    text_health,
)


def _zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _submissions(docs) -> list:
    return group_by_labels(docs, auto_group_labels(docs))


# --- Zips: nested student zips, bombs, nesting depth ---------------------------

def test_nested_student_zips_keep_their_moodle_folder():
    export = _zip({
        "Alice Smith_123456_assignsubmission_file_/sub.zip":
            _zip({"Assignment2.txt": "Alice's own essay " * 40}),
        "Bob Jones_654321_assignsubmission_file_/sub.zip":
            _zip({"Assignment2.txt": "Bob's own essay " * 40}),
    })
    docs = parse_bytes("moodle_export.zip", export)
    assert [d.name for d in docs] == [
        "Alice Smith_123456_assignsubmission_file_/sub/Assignment2.txt",
        "Bob Jones_654321_assignsubmission_file_/sub/Assignment2.txt",
    ]
    subs = _submissions(docs)
    assert len(subs) == 2
    assert "Alice's own essay" in subs[0].text and "Bob" not in subs[0].text


def test_oversized_zip_member_is_skipped_with_a_note():
    buf = io.BytesIO()
    chunk = b"A" * (1 << 20)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("s1111111_report.txt", "a real essay " * 40)
        # Streamed so the test itself never holds the bomb in memory.
        with zf.open("s2222222_bomb.txt", "w") as f:
            for _ in range(dp._MAX_MEMBER_BYTES // len(chunk) + 1):
                f.write(chunk)
    docs = {d.name: d.text for d in parse_bytes("class.zip", buf.getvalue())}
    assert "a real essay" in docs["s1111111_report.txt"]
    bomb = docs["s2222222_bomb.txt"]
    assert len(bomb) < 1000
    assert "too large" in bomb
    assert text_health(bomb) == "read-failed"


def test_zip_total_budget_is_shared_with_nested_zips(monkeypatch):
    monkeypatch.setattr(dp, "_MAX_ZIP_TOTAL_BYTES", 2500)
    member = "word " * 200  # 1000 bytes decompressed
    export = _zip({
        "a.txt": member,
        "sub.zip": _zip({"b.txt": member, "c.txt": member}),
    })
    docs = {d.name: d.text for d in parse_bytes("export.zip", export)}
    assert docs["a.txt"].startswith("word")
    assert docs["sub/b.txt"].startswith("word")
    assert "[Could not read" in docs["sub/c.txt"] and "too large" in docs["sub/c.txt"]


def test_zip_nesting_deeper_than_lms_export_is_refused():
    level3 = _zip({"deep.txt": "should never be opened"})
    level2 = _zip({"inner.zip": level3, "essay.txt": "the student's essay " * 20})
    export = _zip({"Alice_123456_assignsubmission_file_/sub.zip": level2})
    docs = {d.name: d.text for d in parse_bytes("export.zip", export)}
    folder = "Alice_123456_assignsubmission_file_/sub/"
    assert set(docs) == {folder + "essay.txt", folder + "inner.zip"}
    assert "the student's essay" in docs[folder + "essay.txt"]
    assert "[Could not read" in docs[folder + "inner.zip"]
    assert "nested" in docs[folder + "inner.zip"]


# --- Batch upload: same-named files from different uploads must not merge ------

class _Upload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_same_named_files_in_two_student_zips_stay_two_submissions():
    import app

    uploads = [
        _Upload("s1111111.zip", _zip({"report.txt": "first student " * 40})),
        _Upload("s2222222.zip", _zip({"report.txt": "second student " * 40})),
    ]
    docs = app._parse_batch(uploads)
    assert [d.name for d in docs] == [
        "s1111111.zip/report.txt", "s2222222.zip/report.txt",
    ]
    subs = _submissions(docs)
    assert len(subs) == 2
    assert "first student" in subs[0].text and "second" not in subs[0].text


def test_batch_names_without_collision_are_left_alone():
    import app

    uploads = [
        _Upload("s1111111_report.txt", b"first essay " * 40),
        _Upload("class.zip", _zip({"s2222222_report.txt": "second essay " * 40})),
    ]
    assert [d.name for d in app._parse_batch(uploads)] == [
        "s1111111_report.txt", "s2222222_report.txt",
    ]


# --- PDF: counting images must not decode them ---------------------------------

def _pdf_with_images(n: int, image_stream: bytes, filter_: bytes = b"") -> bytes:
    """A one-page PDF whose resources hold `n` 1x1 image XObjects."""
    names = " ".join(f"/Im{i} {4 + i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] "
        f"/Resources << /XObject << {names} >> >> >>".encode(),
    ] + [
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 %s/Length %d >>\n"
        b"stream\n%s\nendstream" % (filter_, len(image_stream), image_stream)
    ] * n
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref,
    )
    return bytes(out)


def test_pdf_images_are_counted_even_when_undecodable():
    pdf = _pdf_with_images(2, b"not a jpeg", filter_=b"/Filter /DCTDecode ")
    text = parse_bytes("scan.pdf", pdf)[0].text
    assert "contains 2 embedded image(s)" in text


def test_pdf_images_are_counted_without_pillow(monkeypatch):
    for mod in ("PIL", "PIL.Image", "pypdf.generic._image_xobject"):
        monkeypatch.setitem(sys.modules, mod, None)
    text = parse_bytes("scan.pdf", _pdf_with_images(1, b"\x00"))[0].text
    assert "contains 1 embedded image(s)" in text
