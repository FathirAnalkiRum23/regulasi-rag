"""Factory function untuk LLM — Gemini atau Ollama dapat ditukar lewat .env."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from regulasi_rag.config import Settings


def get_llm(settings: Settings) -> "BaseChatModel":
    """Kembalikan instance LLM sesuai LLM_PROVIDER di Settings.

    Desain Factory Pattern: kode pemanggil tidak perlu tahu apakah
    memakai Gemini atau Ollama; cukup panggil get_llm(settings).
    Provider baru cukup ditambahkan di sini tanpa mengubah kode lain.
    """
    provider = settings.llm_provider

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        if not settings.google_api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY belum diisi di file .env. "
                "Salin .env.example menjadi .env dan isi API key-nya."
            )
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(
        f"LLM_PROVIDER '{provider}' tidak dikenal. "
        "Gunakan 'gemini' atau 'ollama' di file .env."
    )
