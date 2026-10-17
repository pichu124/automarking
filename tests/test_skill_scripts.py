"""The skill's CLI scripts (intake -> calibrate -> export), run the way the
skill runs them: as subprocesses whose output is piped, which on Windows means
a cp1252 console and cp1252 as the default file encoding."""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import pytest
from docx import Document

from src.calibration import calibrate_scores
from src.config import DEFAULT_CALIBRATION

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "skill" / "rubric-grading" / "scripts"
NAME = "Lê Thị Ngọc"


def run(script, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"  # a piped console on a Windows machine
    p = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, env=env
    )
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("script", ["intake.py", "calibrate.py", "export_docx.py"])
def test_help(script):
    code, out, _ = run(script, "--help")
    assert code == 0 and "usage:" in out


def test_intake_reads_and_writes_unicode(tmp_path):
    src = tmp_path / "in" / f"s4100001_{NAME}_report.docx"
    src.parent.mkdir()
    doc = Document()
    doc.add_paragraph("Hệ số β = 1.2 - Nguyễn")
    doc.save(src)

    code, out, err = run("intake.py", "--work", tmp_path / "work", src)

    assert code == 0, err
    report = tmp_path / "work" / "packs" / "s4100001" / "report.txt"
    assert "Hệ số β = 1.2 - Nguyễn" in report.read_text(encoding="utf-8")
    assert src.name in out


def test_intake_rebuild_drops_files_moved_to_another_student(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    files = {"s4100001_report.txt": "A report", "s4100001_appendix.txt": "B work, misfiled",
             "s4100002_report.txt": "B report"}
    for name, text in files.items():
        (inp / name).write_text(text, encoding="utf-8")
    work = tmp_path / "work"
    assert run("intake.py", "--work", work, *sorted(inp.iterdir()))[0] == 0
    a, b = work / "packs" / "s4100001", work / "packs" / "s4100002"
    assert (a / "report_2.txt").exists()
    (a / "NOTE.txt").write_text("chart image belongs to this student", encoding="utf-8")

    # The professor moves the misfiled appendix to s4100002 by hand (SKILL.md step 2).
    reg_path = work / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    key = next(k for k, i in reg["files"].items() if i["orig"] == "s4100001_appendix.txt")
    reg["files"][key]["sid"] = "s4100002"
    reg["students"]["s4100001"]["files"].remove(key)
    reg["students"]["s4100002"]["files"].append(key)
    reg_path.write_text(json.dumps(reg), encoding="utf-8")
    code, _, err = run("intake.py", "--work", work, inp / "s4100001_report.txt")

    assert code == 0, err
    assert sorted(p.name for p in a.iterdir()) == ["MANIFEST.txt", "NOTE.txt", "report.txt"]
    assert (a / "report.txt").read_text(encoding="utf-8") == "A report"
    assert (b / "report_2.txt").read_text(encoding="utf-8") == "B work, misfiled"


def test_intake_refuses_pack_dir_outside_packs(tmp_path):
    src = tmp_path / "s4100001_report.txt"
    src.write_text("A report", encoding="utf-8")
    work = tmp_path / "work"
    assert run("intake.py", "--work", work, src)[0] == 0
    keep = tmp_path / "report.txt"
    keep.write_text("not intake's file", encoding="utf-8")
    reg_path = work / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg["students"] = {"../..": reg["students"]["s4100001"]}
    reg_path.write_text(json.dumps(reg), encoding="utf-8")

    code, _, err = run("intake.py", "--work", work, src)

    assert code != 0 and "../.." in err
    assert keep.read_text(encoding="utf-8") == "not intake's file"


@pytest.mark.parametrize("n,field,frac,expected", [
    (100, "hd_cap_fraction", 0.29, 29),  # 0.29 * 100 == 28.999999999999996
    (180, "di_cap_fraction", 0.35, 63),  # 0.35 * 180 == 62.99999999999999
])
def test_caps_survive_float_rounding(n, field, frac, expected):
    cfg = dataclasses.replace(DEFAULT_CALIBRATION, **{field: frac})
    _, rep = calibrate_scores([float(i % 100) for i in range(n)], cfg)
    assert (rep.hd_cap if field.startswith("hd") else rep.di_cap) == expected
