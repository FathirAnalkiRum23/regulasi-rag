"""Tahap 1: PDF peraturan -> chunk per Pasal (JSONL) dengan metadata.

Pemakaian:
    python scripts/ingest.py --pdf-dir data/raw --out data/chunks.jsonl

Nama file PDF dipakai sebagai label UU, contoh: uu-27-2022-pdp.pdf
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf

PASAL_RE = re.compile(r"^Pasa[lt]\s+(\d+[A-Z]?)\s*$")
BAB_RE = re.compile(r"^BAB\s+([IVXLC]+)\s*$")
AYAT_RE = re.compile(r"^\(\d+[a-z]?\)\s")
NOISE_RE = re.compile(
    r"^(-\s*\d+\s*-|\d+|www\.[\w.]+|SALINAN|PRESIDEN\s+REPUBLIK\s+INDONESIA|ANOTASI\s+\d{4})$",
    re.I,
)
STOP_RE = re.compile(
    r"^(Disahkan di Jakarta|TAMBAHAN LEMBARAN NEGARA|KEPANITERAAN DAN SEKRETARIAT JENDERAL)\b",
    re.I,
)
MAX_CHARS = 1800


def read_lines(pdf_path: Path) -> list[str]:
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for ln in page.get_text().splitlines():
                ln = ln.strip()
                if ln and not NOISE_RE.match(ln):
                    lines.append(ln)
    return lines


def _split_block_by_size(block: str, max_body: int) -> list[str]:
    """Pecah blok teks per baris agar tetap di bawah batas max_body."""
    chunks, buf = [], ""
    for ln in block.split("\n"):
        if len(ln) > max_body:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(ln), max_body):
                chunks.append(ln[i:i + max_body])
            continue
        candidate = f"{buf}\n{ln}" if buf else ln
        if len(candidate) > max_body and buf:
            chunks.append(buf)
            buf = ln
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def split_long(text: str) -> list[str]:
    """Pecah pasal panjang per ayat; kalau masih panjang, per ukuran."""
    if len(text) <= MAX_CHARS:
        return [text]
    head, *rest = text.split("\n", 1)
    body = rest[0] if rest else ""
    max_body = MAX_CHARS - len(head) - 1

    parts, cur = [], []
    for ln in body.split("\n"):
        if AYAT_RE.match(ln) and cur:
            parts.append("\n".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        parts.append("\n".join(cur))

    expanded = []
    for p in parts:
        if len(p) > max_body:
            expanded.extend(_split_block_by_size(p, max_body))
        else:
            expanded.append(p)

    out, buf = [], ""
    for p in expanded:
        if buf and len(head) + 1 + len(buf) + 1 + len(p) > MAX_CHARS:
            out.append(f"{head}\n{buf}")
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        out.append(f"{head}\n{buf}")
    return out


def chunk_pdf(pdf_path: Path) -> list[dict]:
    uu = pdf_path.stem
    section, bab = "batang_tubuh", ""
    pasal, buf, chunks = None, [], []

    def flush():
        if pasal and buf:
            text = f"Pasal {pasal}\n" + "\n".join(buf)
            for i, part in enumerate(split_long(text)):
                chunks.append({
                    "id": f"{uu}:{section}:pasal-{pasal}:{i}",
                    "uu": uu, "section": section, "bab": bab,
                    "pasal": pasal, "text": part,
                })

    for ln in read_lines(pdf_path):
        if ln.upper() == "PENJELASAN":
            flush(); pasal, buf = None, []
            section, bab = "penjelasan", ""
            continue
        if STOP_RE.match(ln):
            flush(); pasal, buf = None, []
            continue
        if (m := BAB_RE.match(ln)):
            flush(); pasal, buf = None, []
            bab = m.group(1)
            continue
        if (m := PASAL_RE.match(ln)):
            flush()
            pasal, buf = m.group(1), []
            continue
        if pasal:
            buf.append(ln)
    flush()
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/raw")
    ap.add_argument("--out", default="data/chunks.jsonl")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Tidak ada PDF di {args.pdf_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for pdf in pdfs:
            chunks = chunk_pdf(pdf)
            total += len(chunks)
            print(f"{pdf.name}: {len(chunks)} chunk")
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Selesai: {total} chunk -> {args.out}")


if __name__ == "__main__":
    main()
