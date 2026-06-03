"""Document parser resilience: a bad file must never crash the app."""
from __future__ import annotations

from src.document_parser import parse_bytes


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
