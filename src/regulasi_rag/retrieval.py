"""Retrieval lanjutan: hybrid (vektor + BM25 digabung dengan RRF) dan perluasan per bab.

Mode dipilih lewat variabel RETRIEVAL_MODE di file .env:
    vector       baseline: hanya pencarian vektor (default)
    hybrid       vektor + BM25 (dengan stemming bahasa Indonesia), digabung dengan RRF
    hybrid_bab   hybrid, lalu menambahkan pasal lain dari bab yang sama
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

RRF_K = 60          # konstanta standar Reciprocal Rank Fusion
CANDIDATES = 20     # jumlah kandidat dari tiap metode sebelum digabung
BAB_EXTRA = 6       # tambahan maksimum chunk dari perluasan per bab

STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan", "pada", "dalam",
    "ini", "itu", "adalah", "apa", "saja", "bagaimana", "sebagaimana", "dimaksud",
    "pasal", "ayat", "setiap", "dapat", "akan", "oleh", "bagi", "tentang",
}


@lru_cache(maxsize=1)
def _stem_fn():
    """Stemmer bahasa Indonesia (Sastrawi). Kalau belum terpasang, tanpa stemming."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        return StemmerFactory().create_stemmer().stem
    except ImportError:
        return lambda text: text


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", _stem_fn()(text.lower()))
    return [w for w in words if w not in STOPWORDS]


def _is_boilerplate(doc: Document) -> bool:
    """Chunk Penjelasan yang hanya berisi 'Cukup jelas.' tidak berguna untuk jawaban."""
    text = " ".join(doc.page_content.lower().split())
    return len(text) < 70 and "cukup jelas" in text


class _Bm25Index:
    def __init__(self, docs: list[Document]):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi([_tokens(d.page_content) or ["_"] for d in docs])

    def search(self, query: str, n: int) -> list[Document]:
        scores = self.bm25.get_scores(_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.docs[i] for i in order if scores[i] > 0]


def rrf(rankings: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Gabungkan beberapa daftar peringkat dengan Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, 1):
            doc_id = d.metadata["id"]
            by_id[doc_id] = d
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


def expand_by_bab(top: list[Document], all_docs: list[Document], extra: int) -> list[Document]:
    """Jika >= 2 hasil teratas berasal dari bab yang sama, tambahkan pasal lain di bab itu."""
    babs = Counter(
        d.metadata.get("bab")
        for d in top
        if d.metadata.get("section") == "batang_tubuh" and d.metadata.get("bab")
    )
    result = list(top)
    seen = {d.metadata["id"] for d in top}
    added = 0
    for bab, n in babs.most_common():
        if n < 2:
            break
        for d in all_docs:
            if added >= extra:
                return result
            if (
                d.metadata.get("bab") == bab
                and d.metadata.get("section") == "batang_tubuh"
                and d.metadata["id"] not in seen
            ):
                result.append(d)
                seen.add(d.metadata["id"])
                added += 1
    return result


def build_advanced_retriever(settings, embeddings, mode: str):
    """Kembalikan retriever (Runnable: pertanyaan -> daftar Document) untuk mode lanjutan."""
    from regulasi_rag.retriever import _get_vector_store, _load_docs

    vs = _get_vector_store(settings, embeddings)
    all_docs = [d for d in _load_docs(settings.chunks_path) if not _is_boilerplate(d)]
    index = _Bm25Index(all_docs)

    def run(question: str) -> list[Document]:
        vec = [d for d in vs.similarity_search(question, k=CANDIDATES) if not _is_boilerplate(d)]
        lex = index.search(question, CANDIDATES)
        fused = rrf([vec, lex])[: settings.top_k]
        if mode == "hybrid_bab":
            fused = expand_by_bab(fused, all_docs, BAB_EXTRA)
        return fused

    return RunnableLambda(run)
