"""Parse student submissions, group files per student, and build grading packs.

Usage:
    python intake.py --work WORKDIR FILE [FILE ...]

Run it once per upload batch; it keeps a registry in WORKDIR so later batches
merge into the same cohort. Each student gets WORKDIR/packs/<id>/ with:
    report.txt        full text of the Word/PDF report (embedded-image notes kept)
    workbook_N.txt    curated extract of each spreadsheet (every sheet, 60 rows x 40 cols)
    slides.txt        slide text of a .pptx (e.g. Tableau output decks)
    MANIFEST.txt      which source file produced which pack file
Prints a status table and flags unreadable files, students without a
spreadsheet, and files it could not match to a student.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grading_lib.document_parser import parse_bytes, text_health  # noqa: E402

# Student IDs like s4184402 / S4184402 / bare 4184402 (RMIT-style, 7 digits
# starting 3 or 4). The lookbehind stops "Results20230815" matching.
ID_RE = re.compile(r"(?<![A-Za-z0-9])[sS]?([34]\d{6})(?!\d)")
# In the body only an explicit "s"-prefixed ID counts - bare 7-digit numbers
# occur all the time in spreadsheet data.
STRICT_ID_RE = re.compile(r"(?<![A-Za-z0-9])[sS]([34]\d{6})(?!\d)")
DOC_EXTS = ("docx", "pdf", "txt", "md")
SHEET_EXTS = ("xlsx", "xlsm", "xls", "xlsb", "csv")


def _orig_name(path: str) -> str:
    base = os.path.basename(path)
    # Upload tools often prefix a hash: "3a7151b9-Real_Name.docx"
    m = re.match(r"^[0-9a-f]{8}-(.+)$", base)
    return m.group(1) if m else base


def _stem(name: str) -> str:
    s = re.sub(r"\.(docx|pdf|xlsx|xlsm|xls|xlsb|csv|pptx|txt|md)$", "", name.lower())
    s = s.replace("tableau", "").replace("excel", "").replace("calculations", "").replace("report", "")
    return re.sub(r"[^a-z0-9]", "", s)


def _workbook_extract(data: bytes, rows: int = 60, cols: int = 40) -> tuple[str, list]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    out, sheets = [], []
    for ws in wb.worksheets:
        sheets.append(f"{ws.title}({ws.max_row})")
        out.append(f"\n=== Sheet: {ws.title} (max_row={ws.max_row}, max_col={ws.max_column}) ===")
        n = 0
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v)[:28] for v in row[:cols]]
            if any(cells):
                out.append(" | ".join(cells).rstrip(" |")[:360])
                n += 1
            if n >= rows:
                out.append("   ... [rows trimmed - the full sheet is longer]")
                break
    return "\n".join(out), sheets


def _pptx_text(data: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(data))
    slides = sorted(
        (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )
    imgs = len([n for n in z.namelist() if n.startswith("ppt/media/") and not n.endswith("/")])
    body = "\n".join(
        f"{n.split('/')[-1]}: " + " ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8", "replace")))
        for n in slides
    )
    return f"[Slide deck: {len(slides)} slides, {imgs} images (charts/screenshots are images, not text).]\n{body}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    os.makedirs(f"{a.work}/packs", exist_ok=True)
    reg_path = f"{a.work}/registry.json"
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {"files": {}, "students": {}, "unmatched": []}

    for path in a.files:
        key = os.path.abspath(path)
        if key in reg["files"]:
            continue
        orig = _orig_name(path)
        kind = orig.rsplit(".", 1)[-1].lower()
        data = open(path, "rb").read()
        info = {"orig": orig, "kind": kind, "bytes": len(data)}
        text = ""
        if kind == "pptx":
            text = _pptx_text(data)
        else:
            text = parse_bytes(orig, data)[0].text
            info["health"] = text_health(text) if kind in DOC_EXTS else None
        m = ID_RE.search(orig) or (ID_RE.search(text[:3000]) if kind in DOC_EXTS else None) or STRICT_ID_RE.search(text)
        info["sid"] = ("s" + m.group(1)) if m else None
        if kind in DOC_EXTS:
            info["words"] = len(re.sub(r"\[NOTE:[^\]]*\]", "", text).split())
        info["text_cache"] = f"{a.work}/packs/_cache_{abs(hash(key))}.txt"
        if kind in SHEET_EXTS[:4]:
            try:
                text, info["sheets"] = _workbook_extract(data)
            except Exception as exc:  # fall back to the parser's text
                info["sheets"] = [f"ERR {exc}"]
        open(info["text_cache"], "w").write(text)
        reg["files"][key] = info
        if info["sid"]:
            reg["students"].setdefault(info["sid"], {"files": []})["files"].append(key)
        else:
            reg["unmatched"].append(key)

    # Files without an ID join the student whose file shares their name stem.
    still = []
    for key in reg["unmatched"]:
        st = _stem(reg["files"][key]["orig"])
        hit = next(
            (k for k, i in reg["files"].items() if i["sid"] and k != key and st
             and (_stem(i["orig"]) == st or (len(st) > 10 and (st in _stem(i["orig"]) or _stem(i["orig"]) in st)))),
            None,
        )
        if hit:
            sid = reg["files"][hit]["sid"]
            reg["files"][key]["sid"] = sid
            reg["students"][sid]["files"].append(key)
        else:
            still.append(key)
    # ID-less files that share a stem with each other are one student,
    # keyed by name until the professor supplies the ID.
    by_stem = {}
    for key in still:
        by_stem.setdefault(_stem(reg["files"][key]["orig"]), []).append(key)
    still = []
    for st, keys in by_stem.items():
        if len(keys) > 1 and st:
            sid = "name-" + st[:40]
            for key in keys:
                reg["files"][key]["sid"] = sid
            reg["students"].setdefault(sid, {"files": []})["files"].extend(keys)
        else:
            still.extend(keys)
    reg["unmatched"] = still

    # Rebuild every student's pack from the cache.
    for sid, st in reg["students"].items():
        d = f"{a.work}/packs/{sid}"
        os.makedirs(d, exist_ok=True)
        man, nwb, ndoc, nppt = [], 0, 0, 0
        for key in st["files"]:
            i = reg["files"][key]
            t = open(i["text_cache"]).read()
            if i["kind"] in DOC_EXTS:
                ndoc += 1
                name = "report.txt" if ndoc == 1 else f"report_{ndoc}.txt"
            elif i["kind"] == "pptx":
                nppt += 1
                name = "slides.txt" if nppt == 1 else f"slides_{nppt}.txt"
            else:
                nwb += 1
                name = f"workbook_{nwb}.txt"
            open(f"{d}/{name}", "w").write(t)
            man.append(f"{name} <- {i['orig']}")
        open(f"{d}/MANIFEST.txt", "w").write("\n".join(man))

    json.dump(reg, open(reg_path, "w"), indent=1)
    print(f"Students: {len(reg['students'])}")
    for sid, st in sorted(reg["students"].items()):
        fs = [reg["files"][k] for k in st["files"]]
        kinds = [f["kind"] for f in fs]
        flags = []
        if not any(k in SHEET_EXTS for k in kinds):
            flags.append("NO SPREADSHEET")
        if not any(k in DOC_EXTS for k in kinds):
            flags.append("NO REPORT")
        flags += [f"{f['orig']}: {f['health']}" for f in fs if f.get("health")]
        words = [f.get("words") for f in fs if f.get("words")]
        print(f"- {sid}: {', '.join(f['orig'] for f in fs)} | words~{words}" + (f" | FLAGS: {flags}" if flags else ""))
    for key in reg["unmatched"]:
        print("UNMATCHED:", reg["files"][key]["orig"])


if __name__ == "__main__":
    main()
