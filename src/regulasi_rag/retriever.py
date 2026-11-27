"""Embedding dan retrieval vektor dari Chroma."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever

from regulasi_rag.config import Settings


def get_embeddings(settings: Settings):
    """Kembalikan model embedding HuggingFace sesuai config.

    Model di-cache ke HF_HOME (default D:\\.cache\\huggingface) agar
    tidak diunduh ulang setiap kali program dijalankan.
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store(settings: Settings, embeddings):
    """Kembalikan instance Chroma yang terhubung ke direktori lokal."""
    from langchain_chroma import Chroma  # type: ignore

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def _load_docs(chunks_path: Path) -> list:
    """Baca chunks.jsonl menjadi daftar Document LangChain."""
    from langchain_core.documents import Document  # type: ignore

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"File {chunks_path} tidak ditemukan. "
            "Jalankan scripts/ingest.py terlebih dahulu."
        )

    docs = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            docs.append(
                Document(
                    page_content=c["text"],
                    metadata={
                        "id": c["id"],
                        "uu": c["uu"],
                        "section": c["section"],
                        "bab": c["bab"],
                        "pasal": c["pasal"],
                    },
                )
            )
    return docs


def build_index(settings: Settings, embeddings) -> int:
    """Baca data/chunks.jsonl dan masukkan seluruh chunk ke Chroma.

    Hanya dijalankan sekali (atau ulang jika chunks.jsonl berubah).
    Mengembalikan jumlah chunk yang berhasil diindeks.
    """
    docs = _load_docs(settings.chunks_path)

    # Pengaman 1: ID harus unik, kalau tidak chunk bisa tertimpa diam-diam.
    dup = [i for i, n in Counter(d.metadata["id"] for d in docs).items() if n > 1]
    if dup:
        raise ValueError(f"ID chunk duplikat ({len(dup)}), contoh: {dup[:3]}")

    vs = _get_vector_store(settings, embeddings)
    # Hapus koleksi lama agar tidak ada duplikat saat re-index.
    vs.delete_collection()
    vs = _get_vector_store(settings, embeddings)

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d.metadata["id"] for d in batch]
        vs.add_documents(batch, ids=ids)
        print(f"  Diindeks {min(i + batch_size, len(docs))}/{len(docs)} chunk...", end="\r")

    # Pengaman 2: jumlah di Chroma harus sama dengan jumlah chunk.
    n = len(vs.get()["ids"])
    if n != len(docs):
        raise RuntimeError(f"Terindeks {n} dari {len(docs)} chunk. Cek ID duplikat.")

    print(f"\nIndexing selesai: {len(docs)} chunk -> {settings.chroma_dir}")
    return len(docs)


def get_retriever(settings: Settings, embeddings):
    """Kembalikan retriever sesuai RETRIEVAL_MODE (vector, hybrid, atau hybrid_bab)."""
    mode = os.getenv("RETRIEVAL_MODE", "vector").strip().lower()
    if mode in ("hybrid", "hybrid_bab"):
        from regulasi_rag.retrieval import build_advanced_retriever

        return build_advanced_retriever(settings, embeddings, mode)
    vs = _get_vector_store(settings, embeddings)
    return vs.as_retriever(search_kwargs={"k": settings.top_k})
