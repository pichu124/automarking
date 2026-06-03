"""Extract plain text from uploaded documents.

Supports .docx, .pdf, .txt/.md, and .zip archives (which are expanded into one
entry per contained document). Everything returned is plain text keyed by a
display name, which downstream code feeds to the LLM.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import List

import docx  # python-docx
from pypdf import PdfReader


_TEXT_EXTS = (".txt", ".md", ".csv")
_SKIP_PREFIXES = ("__MACOSX/", ".")


@dataclass
class ParsedDocument:
    name: str
    text: str


def _read_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    # Include simple table text — rubrics and grade sheets are often tables.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def parse_bytes(name: str, data: bytes) -> List[ParsedDocument]:
    """Parse one uploaded file. A .zip expands to multiple ParsedDocuments."""
    lowered = name.lower()
    if lowered.endswith(".zip"):
        return _parse_zip(data)
    if lowered.endswith(".docx"):
        return [ParsedDocument(name, _read_docx(data))]
    if lowered.endswith(".pdf"):
        return [ParsedDocument(name, _read_pdf(data))]
    if lowered.endswith(_TEXT_EXTS):
        return [ParsedDocument(name, _read_text(data))]
    # Unknown extension: best-effort decode so nothing is silently dropped.
    return [ParsedDocument(name, _read_text(data))]


def _parse_zip(data: bytes) -> List[ParsedDocument]:
    out: List[ParsedDocument] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = info.filename.rsplit("/", 1)[-1]
            if info.filename.startswith(_SKIP_PREFIXES) or base.startswith("."):
                continue
            inner = zf.read(info)
            try:
                out.extend(parse_bytes(info.filename, inner))
            except Exception as exc:  # one bad file shouldn't sink the batch
                out.append(
                    ParsedDocument(info.filename, f"[Could not read file: {exc}]")
                )
    return out


def parse_upload(uploaded_file) -> List[ParsedDocument]:
    """Parse a Streamlit UploadedFile (or anything with .name and .getvalue())."""
    return parse_bytes(uploaded_file.name, uploaded_file.getvalue())
