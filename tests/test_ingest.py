"""Unit test deterministik untuk logika chunking di scripts/ingest.py."""

import importlib.util
from pathlib import Path

import pymupdf


def _load_ingest_module():
    root = Path(__file__).resolve().parents[1]
    ingest_path = root / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("ingest", ingest_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_module()


def test_split_long_keeps_short_text_single_chunk():
    text = "Pasal 1\nKetentuan umum singkat."
    assert ingest.split_long(text) == [text]


def test_split_long_splits_by_ayat_and_preserves_pasal_header():
    ayat_1 = "(1) " + ("kalimat satu " * 80).strip()
    ayat_2 = "(2) " + ("kalimat dua " * 80).strip()
    ayat_3 = "(3) " + ("kalimat tiga " * 80).strip()
    text = f"Pasal 10\n{ayat_1}\n{ayat_2}\n{ayat_3}"

    parts = ingest.split_long(text)

    assert len(parts) >= 2
    assert all(part.startswith("Pasal 10\n") for part in parts)
    assert all(len(part) <= ingest.MAX_CHARS for part in parts)


def test_chunk_pdf_extracts_sections_metadata_and_strips_noise(tmp_path):
    pdf_path = tmp_path / "uu-contoh.pdf"
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "\n".join([
            "SALINAN",
            "PRESIDEN REPUBLIK INDONESIA",
            "BAB I",
            "Pasal 1",
            "(1) Definisi pertama.",
            "(2) Definisi kedua.",
            "Pasal 2",
            "Ketentuan asas.",
            "- 1 -",
        ]),
    )

    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "\n".join([
            "PENJELASAN",
            "Pasal 1",
            "Cukup jelas.",
            "www.peraturan.go.id",
        ]),
    )

    doc.save(pdf_path)
    doc.close()

    chunks = ingest.chunk_pdf(pdf_path)

    assert len(chunks) == 3
    assert chunks[0]["id"] == "uu-contoh:batang_tubuh:pasal-1:0"
    assert chunks[0]["uu"] == "uu-contoh"
    assert chunks[0]["section"] == "batang_tubuh"
    assert chunks[0]["bab"] == "I"
    assert chunks[0]["pasal"] == "1"
    assert "SALINAN" not in chunks[0]["text"]
    assert "PRESIDEN REPUBLIK INDONESIA" not in chunks[0]["text"]
    assert "- 1 -" not in chunks[0]["text"]

    assert chunks[1]["id"] == "uu-contoh:batang_tubuh:pasal-2:0"
    assert chunks[1]["pasal"] == "2"

    assert chunks[2]["id"] == "uu-contoh:penjelasan:pasal-1:0"
    assert chunks[2]["section"] == "penjelasan"
    assert chunks[2]["bab"] == ""
    assert "www.peraturan.go.id" not in chunks[2]["text"]
