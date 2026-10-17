"""Drift guard: the packaged skill ships copies of src/ modules and a zipped
.skill bundle. A fix in src/ that is not copied over silently leaves the skill
grading with the old code, so fail loudly instead."""
from __future__ import annotations

import pathlib
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKILL = ROOT / "skill"
LIB = SKILL / "rubric-grading" / "scripts" / "grading_lib"
SHARED = ["calibration", "config", "document_parser", "docx_export", "models"]


def _text(data: bytes) -> bytes:
    # git autocrlf rewrites line endings on Windows checkouts; compare content.
    return data.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("name", SHARED)
def test_grading_lib_matches_src(name):
    src = ROOT / "src" / f"{name}.py"
    copy = LIB / f"{name}.py"
    assert _text(src.read_bytes()) == _text(copy.read_bytes()), (
        f"{copy.relative_to(ROOT)} drifted from src/{name}.py - copy it over "
        "and rebuild skill/rubric-grading.skill"
    )


def test_skill_bundle_matches_folder():
    with zipfile.ZipFile(SKILL / "rubric-grading.skill") as bundle:
        stale = [
            n for n in bundle.namelist()
            if not n.endswith("/")
            and _text(bundle.read(n)) != _text((SKILL / n).read_bytes())
        ]
        packed = {n for n in bundle.namelist() if not n.endswith("/")}
    on_disk = {
        p.relative_to(SKILL).as_posix()
        for p in (SKILL / "rubric-grading").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert not stale, f"rebuild skill/rubric-grading.skill; stale: {stale}"
    assert packed == on_disk, f"bundle/folder file sets differ: {packed ^ on_disk}"
