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
    parts: List[str] = []
    for page in reader.pages:
        # "plain" mode avoids pypdf's layout-mode width maths, which can crash
        # on PDFs whose font-width entries are unresolved indirect objects
        # (TypeError: unsupported operand type(s) for <: 'float' and ...).
        # A per-page guard means one bad page never sinks the whole document.
        try:
            parts.append(page.extract_text(extraction_mode="plain") or "")
        except TypeError:  # older pypdf without the kwarg (already plain)
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        except Exception:
            parts.append("")
    return "\n".join(parts)


def _read_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def parse_bytes(name: str, data: bytes) -> List[ParsedDocument]:
    """Parse one uploaded file. A .zip expands to multiple ParsedDocuments."""
    lowered = name.lower()
    if lowered.endswith(".zip"):
        return _parse_zip(data)
    # A single unreadable file must never crash the app — surface a placeholder.
    try:
        if lowered.endswith(".docx"):
            text = _read_docx(data)
        elif lowered.endswith(".pdf"):
            text = _read_pdf(data)
        elif lowered.endswith(_TEXT_EXTS):
            text = _read_text(data)
        else:
            # Unknown extension: best-effort decode so nothing is silently dropped.
            text = _read_text(data)
    except Exception as exc:
        text = f"[Could not read {name}: {exc}]"
    return [ParsedDocument(name, text)]


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
