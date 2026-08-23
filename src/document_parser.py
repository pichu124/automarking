"""Extract plain text from uploaded documents.

Supports .docx, .pdf, .txt/.md, and .zip archives (which are expanded into one
entry per contained document). Everything returned is plain text keyed by a
display name, which downstream code feeds to the LLM.
"""
from __future__ import annotations

import html
import io
import re
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
    text = "\n".join(parts)
    if text.strip():
        return text
    # Fallback: python-docx's paragraph/table API misses text boxes, shapes,
    # headers and footers. Pull text straight from the docx XML so rubrics laid
    # out in a text box still come through.
    return _docx_xml_text(data)


def _docx_xml_text(data: bytes) -> str:
    """Extract text from a docx's XML parts (catches text boxes/headers)."""
    out: List[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [
            n for n in zf.namelist()
            if n.startswith("word/") and n.endswith(".xml")
            and any(k in n for k in ("document", "header", "footer"))
        ]
        names.sort(key=lambda n: (0 if n == "word/document.xml" else 1, n))
        for name in names:
            xml = zf.read(name).decode("utf-8", errors="replace")
            # Preserve structure, then strip tags so text-box runs survive.
            xml = xml.replace("</w:p>", "\n").replace("<w:tab/>", "\t")
            xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
            stripped = re.sub(r"<[^>]+>", "", xml)
            cleaned = html.unescape(stripped).strip()
            if cleaned:
                out.append(cleaned)
    return "\n".join(out).strip()


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
    """Decode text robustly: Excel-exported CSVs are often UTF-8-with-BOM,
    UTF-16 or cp1252 rather than plain UTF-8."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeError:
            pass
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_binary(text: str) -> bool:
    """True when decoded 'text' is really binary garbage (mojibake guard)."""
    sample = text[:4000]
    if not sample:
        return False
    bad = sum(
        1 for ch in sample
        if ch == "�" or (ord(ch) < 32 and ch not in "\t\n\r")
    )
    return bad > len(sample) * 0.10


def _read_xlsx(data: bytes) -> str:
    """Extract text from an Excel workbook, one row per line, sheets labelled.

    Rubrics and grade sheets often live in Excel. ``data_only=True`` returns
    computed cell values instead of formula strings.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'openpyxl' package is required to read Excel files. "
            "Install it with: pip install openpyxl"
        ) from exc

    workbook = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    parts: List[str] = []
    try:
        for sheet in workbook.worksheets:
            if len(workbook.worksheets) > 1:
                parts.append(f"=== Sheet: {sheet.title} ===")
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if v is None else str(v).strip() for v in row]
                if any(cells):
                    parts.append(" | ".join(cells).rstrip(" |"))
    finally:
        workbook.close()
    return "\n".join(parts)


def _read_xlsb(data: bytes) -> str:
    """Extract text from an Excel Binary Workbook (.xlsb) via pyxlsb."""
    try:
        from pyxlsb import open_workbook
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'pyxlsb' package is required to read .xlsb files. "
            "Install it with: pip install pyxlsb"
        ) from exc
    parts: List[str] = []
    with open_workbook(io.BytesIO(data)) as wb:
        for sheet_name in wb.sheets:
            with wb.get_sheet(sheet_name) as sheet:
                rows_out: List[str] = []
                for row in sheet.rows():
                    cells = ["" if c.v is None else str(c.v).strip() for c in row]
                    if any(cells):
                        rows_out.append(" | ".join(cells).rstrip(" |"))
                if rows_out:
                    if len(wb.sheets) > 1:
                        parts.append(f"=== Sheet: {sheet_name} ===")
                    parts.extend(rows_out)
    return "\n".join(parts)


def _read_xls(data: bytes) -> str:
    """Extract text from a legacy Excel 97-2003 workbook (.xls) via xlrd."""
    try:
        import xlrd
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'xlrd' package is required to read legacy .xls files. "
            "Install it with: pip install xlrd"
        ) from exc
    book = xlrd.open_workbook(file_contents=data)
    parts: List[str] = []
    for sheet in book.sheets():
        rows_out: List[str] = []
        for ri in range(sheet.nrows):
            cells = [
                "" if v in ("", None) else str(v).strip()
                for v in sheet.row_values(ri)
            ]
            if any(cells):
                rows_out.append(" | ".join(cells).rstrip(" |"))
        if rows_out:
            if book.nsheets > 1:
                parts.append(f"=== Sheet: {sheet.name} ===")
            parts.extend(rows_out)
    return "\n".join(parts)


# Embedded images (screenshots, charts) cannot be text-extracted. Count them
# and prepend a note so the grader never mistakes invisible-for-absent — a
# student's appendix full of rate screenshots must not be called "empty".
_IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".emf", ".wmf", ".webp",
)


def _count_opc_images(data: bytes, prefix: str) -> int:
    """Count image files inside an OPC (zip) container under `prefix`.

    Works for Word (word/media/) and every Excel flavour that is a zip —
    .xlsx, .xlsm and .xlsb all store pasted images (e.g. Solver-dialog
    screenshots) as ordinary files under xl/media/.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return sum(
                1 for n in zf.namelist()
                if n.lower().startswith(prefix) and n.lower().endswith(_IMAGE_EXTS)
            )
    except Exception:
        return 0


def _count_docx_images(data: bytes) -> int:
    return _count_opc_images(data, "word/media/")


def _count_xl_images(data: bytes) -> int:
    return _count_opc_images(data, "xl/media/")


def _count_pdf_images(data: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(data))
        total = 0
        for page in reader.pages:
            try:
                total += sum(1 for _ in page.images)
            except Exception:
                continue
        return total
    except Exception:
        return 0


def _image_note(count: int) -> str:
    if count <= 0:
        return ""
    return (
        f"[NOTE: this document contains {count} embedded image(s) - e.g. "
        "screenshots, charts, figures - whose content cannot be extracted as "
        "text here. Do NOT state that screenshots, figures or appendix "
        "evidence are missing or absent; they may be in these images. If "
        "required evidence is not visible in the text, say it could not be "
        "verified from the text and grade what is verifiable.]\n\n"
    )


def _embedded_image_note(lowered: str, data: bytes) -> str:
    """The image-count note for any container format that can embed images."""
    if lowered.endswith(".docx"):
        return _image_note(_count_docx_images(data))
    if lowered.endswith((".xlsx", ".xlsm", ".xlsb")):
        return _image_note(_count_xl_images(data))
    if lowered.endswith(".pdf"):
        return _image_note(_count_pdf_images(data))
    return ""


def parse_bytes(name: str, data: bytes) -> List[ParsedDocument]:
    """Parse one uploaded file. A .zip expands to multiple ParsedDocuments."""
    lowered = name.lower()
    if lowered.endswith(".zip"):
        return _parse_zip(data)

    # A standalone image (e.g. a screenshot submitted as its own file) has no
    # extractable text — surface an explicit note instead of binary garbage, so
    # after per-student grouping the grader knows the image evidence EXISTS.
    if lowered.endswith(_IMAGE_EXTS):
        return [ParsedDocument(name, (
            "[Image file - visual content (e.g. a screenshot) that cannot be "
            "extracted as text. Do NOT treat the evidence it may contain as "
            "missing or absent; say it could not be verified from the text.]"
        ))]

    # A single unreadable file must never crash the app — surface a placeholder,
    # keeping the embedded-image note even when the body fails to parse.
    note = _embedded_image_note(lowered, data)
    try:
        if lowered.endswith(".docx"):
            text = note + _read_docx(data)
        elif lowered.endswith(".pdf"):
            text = note + _read_pdf(data)
        elif lowered.endswith((".xlsx", ".xlsm")):
            text = note + _read_xlsx(data)
        elif lowered.endswith(".xlsb"):
            text = note + _read_xlsb(data)
        elif lowered.endswith(".xls"):
            text = note + _read_xls(data)
        elif lowered.endswith(_TEXT_EXTS):
            text = _read_text(data)
        else:
            # Unknown extension: best-effort decode, but never feed the model
            # silent mojibake from an unsupported binary format.
            text = _read_text(data)
            if _looks_binary(text):
                text = (
                    f"[Unsupported binary file format ({name}): text could not "
                    "be extracted. Convert it to docx/pdf/xlsx/csv if its "
                    "content should be graded; do not treat its content as "
                    "missing.]"
                )
    except Exception as exc:
        text = note + f"[Could not read {name}: {exc}]"
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


# --- Grouping files into per-student submissions ---------------------------
# A submission often spans several files (a Word report + an Excel workbook).
# Group by student ID in the filename (e.g. s4180007); files without an ID
# group by their normalised filename stem, so "X.docx" pairs with "X.xlsx".

_STUDENT_ID_RE = re.compile(r"[sS](\d{6,8})")


def _group_key(name: str) -> tuple:
    base = name.rsplit("/", 1)[-1]
    match = _STUDENT_ID_RE.search(base)
    if match:
        return ("id", match.group(1))
    stem = base.rsplit(".", 1)[0].lower()
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)      # drop copy suffixes " (1)"
    stem = re.sub(r"[^a-z0-9]+", "", stem)           # normalise punctuation
    return ("stem", stem or base.lower())


def group_by_student(
    docs: List[ParsedDocument], enabled: bool = True
) -> List[ParsedDocument]:
    """Merge files that belong to the same student into one submission.

    The primary (Word/PDF) file's name labels the group and its text comes
    first, so if a length cap ever trims the combined text it is the auxiliary
    spreadsheets that get shortened, not the write-up.
    """
    if not enabled:
        return list(docs)
    groups: dict = {}
    order: List[tuple] = []
    for doc in docs:
        key = _group_key(doc.name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(doc)

    merged: List[ParsedDocument] = []
    for key in order:
        members = groups[key]
        if len(members) == 1:
            merged.append(members[0])
            continue
        primary = next(
            (m for m in members if m.name.lower().endswith((".docx", ".pdf"))),
            members[0],
        )
        ordered = [primary] + [m for m in members if m is not primary]
        text = "\n\n".join(
            f"=== File: {m.name} ===\n{m.text.strip()}" for m in ordered
        )
        label = f"{primary.name} [+{len(members) - 1} file(s)]"
        merged.append(ParsedDocument(label, text))
    return merged
