"""RAG chain: retrieval + generasi jawaban dengan sitasi pasal."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

SYSTEM_PROMPT = """\
Kamu adalah asisten pembaca UU PDP (Undang-Undang Nomor 27 Tahun 2022 tentang \
Pelindungan Data Pribadi). Jawab dalam Bahasa Indonesia yang jelas dan ringkas.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan teks pasal yang diberikan di bawah sebagai KONTEKS.
2. Setiap klaim dalam jawaban HARUS disertai sitasi nomor pasal (contoh: "Pasal 5", \
"Pasal 12 ayat (2)").
3. Setiap potongan konteks diberi label BATANG TUBUH atau PENJELASAN. BATANG TUBUH \
adalah bunyi pasal yang mengikat. PENJELASAN hanya keterangan tambahan: jika kamu \
memakainya, tulis sitasinya sebagai "Penjelasan Pasal N", jangan sebagai "Pasal N".
4. Jika konteks tidak memuat informasi yang cukup untuk menjawab pertanyaan, atau \
pertanyaannya tidak berkaitan dengan UU PDP, tolak dengan jelas: "Maaf, tidak ada \
dasar pasal yang cukup untuk menjawab pertanyaan ini berdasarkan UU PDP."
5. Jangan mengarang, menginterpretasi, atau menambahkan informasi di luar konteks. \
Teks konteks kadang mengandung salah ketik akibat OCR; jangan menebak bunyi yang \
tidak terbaca jelas.
6. Ini adalah alat bantu membaca undang-undang, BUKAN nasihat hukum.

KONTEKS (teks pasal):
{context}
"""

HUMAN_PROMPT = "Pertanyaan: {question}"


def _format_docs(docs) -> str:
    """Gabungkan dokumen-dokumen retrieval menjadi satu blok konteks."""
    parts = []
    for d in docs:
        section = "PENJELASAN" if d.metadata.get("section") == "penjelasan" else "BATANG TUBUH"
        label = f"[Pasal {d.metadata.get('pasal', '?')} | {section}]"
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


def build_chain(llm, retriever):
    """Bangun RAG chain: retrieve -> format konteks -> generate -> parse.

    Mengembalikan chain yang menerima string pertanyaan dan menghasilkan
    string jawaban. Dipakai oleh CLI (ask.py).
    """
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _build_prompt()
        | llm
        | StrOutputParser()
    )
    return chain


def answer_with_sources(llm, retriever, question: str) -> dict:
    """Jawab pertanyaan dan kembalikan juga chunk sumbernya.

    Dipakai untuk evaluasi (hit@k, MRR, RAGAS) dan antarmuka, karena
    membutuhkan daftar pasal dan teks yang benar-benar diambil retriever.
    """
    docs = retriever.invoke(question)
    answer = (_build_prompt() | llm | StrOutputParser()).invoke(
        {"context": _format_docs(docs), "question": question}
    )
    sources = [
        {
            "id": d.metadata.get("id"),
            "pasal": d.metadata.get("pasal"),
            "section": d.metadata.get("section"),
            "text": d.page_content,
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}