"""Tests for the Phase-6 parse fallback heuristics (etl/parse_fallback.py) and the copy in
etl/upload_ingest.py.

    uv run --no-project --with-requirements eval/requirements.txt --with pymupdf pytest -q etl/test_parse_fallback.py
"""

from __future__ import annotations

import pytest

import parse_fallback as pf
import upload_ingest as ui

CLEAN_FR = "La politique nationale de lutte contre le cancer prévoit un dépistage précoce chez l'enfant."
CLEAN_HI = "राष्ट्रीय कैंसर नियंत्रण कार्यक्रम बच्चों में कैंसर की शीघ्र पहचान पर केंद्रित है।"
CID = "Ra(cid:12)(cid:7)tr(cid:44)ya ka(cid:3)nsar"


@pytest.mark.parametrize("text,language,reason", [
    (CLEAN_FR, "french", None),
    (CLEAN_HI, "hindi", None),
    ("", "english", "empty"),
    ("   \n ", "english", "empty"),
    (CID, "english", "cid_markers"),
    ("abc��" + "x" * 50, "english", "replacement_chars"),  # 2/56 > 2%
    ("Rashtriya cancer niyantran karyakram", "hindi", "missing_script"),  # transliterated / broken text layer
    ("Rashtriya cancer niyantran karyakram", "english", None),
])
def test_fallback_reason(text, language, reason):
    assert pf.fallback_reason(text, language) == reason
    assert ui.fallback_reason(text, language) == reason  # the upload_ingest copy agrees


@pytest.mark.parametrize("original,fitz_text,language,expected", [
    (CID, CLEAN_FR, "french", True),          # garbled -> clean
    ("Rashtriya cancer", CLEAN_HI, "hindi", True),  # PyMuPDF recovers Devanagari
    ("", CLEAN_FR, "french", True),           # empty -> text
    (CLEAN_FR, CLEAN_FR + " x", "french", False),  # original fine
    (CID, "", "french", False),               # PyMuPDF got nothing
    (CID, CID, "french", False),              # both garbled
])
def test_prefer_fitz(original, fitz_text, language, expected):
    assert pf.prefer_fitz(original, fitz_text, language) is expected
    assert ui.prefer_fitz(original, fitz_text, language) is expected


def test_blocks_to_elements_detects_headers_and_running_sections():
    pages = [[("1. Objectifs", 16.0), (CLEAN_FR, 10.0), (CLEAN_FR, 10.0)],
             [("2. Budget\n", 16.0), ("Le budget  annuel\nest de 3 M.", 10.0)]]
    rows = pf.blocks_to_elements(pages)
    assert rows == ui.blocks_to_elements(pages)
    assert [(r["idx"], r["elem_type"], r["page"], r["section"]) for r in rows] == [
        (0, "section_header", 0, "1. Objectifs"), (1, "text", 0, "1. Objectifs"), (2, "text", 0, "1. Objectifs"),
        (3, "section_header", 1, "2. Budget"), (4, "text", 1, "2. Budget"),
    ]
    assert rows[4]["content"] == "Le budget annuel est de 3 M." and rows[4]["header_text"] is None


def test_long_or_body_size_blocks_are_not_headers():
    rows = pf.blocks_to_elements([[("x " * 100, 20.0), ("short", 10.0), ("body", 10.0)]])
    assert [r["elem_type"] for r in rows] == ["text", "text", "text"]
    assert pf.blocks_to_elements([]) == []


def test_pdf_blocks_reads_a_real_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "policy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Objectives", fontsize=18)
    page.insert_text((72, 120), "The plan screens children for cancer early.", fontsize=10)
    doc.save(path)
    doc.close()
    rows = pf.blocks_to_elements(pf.pdf_blocks(str(path)))
    assert [(r["elem_type"], r["content"]) for r in rows] == [
        ("section_header", "Objectives"), ("text", "The plan screens children for cancer early.")]
    assert rows[1]["section"] == "Objectives"


def test_upload_with_parse_fallback_keeps_clean_docs(monkeypatch):
    elements = [{"doc_id": "d1", "idx": 0, "elem_type": "text", "page": 0, "content": CLEAN_FR, "section": "(no section)"},
                {"doc_id": "d2", "idx": 0, "elem_type": "text", "page": 0, "content": CID, "section": "(no section)"}]
    monkeypatch.setattr(ui, "pdf_blocks", lambda path: [[("Objectifs", 16.0), (CLEAN_FR, 10.0)]])
    out = ui.with_parse_fallback(elements, [("d1", "/Volumes/a.pdf", "a.pdf"), ("d2", "/Volumes/b.pdf", "b.pdf")], "french")
    assert [e for e in out if e["doc_id"] == "d1"] == elements[:1]
    d2 = [e for e in out if e["doc_id"] == "d2"]
    assert [(e["elem_type"], e["section"]) for e in d2] == [("section_header", "Objectifs"), ("text", "Objectifs")]
    assert set(d2[0]) == {"doc_id", "idx", "elem_type", "page", "content", "section"}
