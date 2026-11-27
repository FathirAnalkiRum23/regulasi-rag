"""CLI entry point: python -m regulasi_rag.ask "pertanyaan".

Pemakaian:
    python -m regulasi_rag.ask "Apa kewajiban Pengendali Data Pribadi?"
    python -m regulasi_rag.ask "Apa kewajiban Pengendali Data Pribadi?" -v
    python -m regulasi_rag.ask --index          (hanya bangun ulang index)
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    # Pastikan teks Indonesia tampil benar di konsol Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        prog="python -m regulasi_rag.ask",
        description="Tanya-jawab UU PDP berbasis RAG dengan sitasi pasal.",
    )
    ap.add_argument(
        "question",
        nargs="?",
        help='Pertanyaan, contoh: "Hak apa saja yang dimiliki Subjek Data Pribadi?"',
    )
    ap.add_argument(
        "--index",
        action="store_true",
        help="Bangun ulang index Chroma dari data/chunks.jsonl (boleh tanpa pertanyaan).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Jumlah chunk yang diambil (override TOP_K di .env).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Tampilkan cuplikan teks tiap chunk yang diambil retriever.",
    )
    args = ap.parse_args(argv)

    if not args.question and not args.index:
        ap.error("isi pertanyaan, atau pakai --index untuk membangun ulang index.")

    from regulasi_rag.config import get_settings
    from regulasi_rag.llm import get_llm
    from regulasi_rag.retriever import build_index, get_embeddings, get_retriever
    from regulasi_rag.chain import answer_with_sources

    settings = get_settings()
    if args.top_k is not None:
        import dataclasses

        settings = dataclasses.replace(settings, top_k=args.top_k)

    print(f"[info] LLM: {settings.llm_provider}/{settings.llm_model}")
    print(f"[info] Embedding: {settings.embedding_model} ({settings.embedding_device})")

    print("[info] Memuat model embedding (pertama kali bisa memakan waktu)...")
    embeddings = get_embeddings(settings)

    chroma_exists = settings.chroma_dir.exists() and any(settings.chroma_dir.iterdir())

    if args.index or not chroma_exists:
        print("[info] Membangun index Chroma dari data/chunks.jsonl...")
        n = build_index(settings, embeddings)
        print(f"[info] {n} chunk diindeks.")
    else:
        print(f"[info] Memakai index yang sudah ada di {settings.chroma_dir}")

    if not args.question:
        return

    retriever = get_retriever(settings, embeddings)
    llm = get_llm(settings)

    print(f"\n{'=' * 60}")
    print(f"Pertanyaan: {args.question}")
    print(f"{'=' * 60}\n")

    result = answer_with_sources(llm, retriever, args.question)
    print(result["answer"])

    print(f"\n{'-' * 60}")
    print("Chunk yang diambil retriever:")
    for i, s in enumerate(result["sources"], 1):
        label = "Penjelasan" if s["section"] == "penjelasan" else "Batang tubuh"
        print(f"  {i}. Pasal {s['pasal']} ({label})")
        if args.verbose:
            snippet = " ".join(s["text"].split())[:150]
            print(f"     {snippet}...")

    print(f"\n{'=' * 60}")
    print("DISCLAIMER: Ini alat bantu baca UU, bukan nasihat hukum.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()