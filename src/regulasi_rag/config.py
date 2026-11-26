"""Konfigurasi terpusat yang membaca variabel lingkungan dan file .env."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    """Konversi nilai string env var ke boolean secara aman."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    """Ubah path relatif menjadi absolut terhadap akar project."""
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


@dataclass(frozen=True)
class Settings:
    """Menyimpan seluruh konfigurasi aplikasi tanpa nilai rahasia hardcoded."""

    project_root: Path
    llm_provider: str
    llm_model: str
    llm_temperature: float
    ollama_base_url: str
    embedding_model: str
    embedding_device: str
    top_k: int
    use_reranker: bool
    reranker_model: str
    chunks_path: Path
    chroma_dir: Path
    chroma_collection: str
    # repr=False mencegah API key tercetak secara tidak sengaja di log/traceback.
    google_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, project_root: Path = PROJECT_ROOT) -> "Settings":
        """Bangun Settings dari variabel lingkungan yang aktif."""
        api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        return cls(
            project_root=project_root,
            llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
            llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash").strip(),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
            top_k=int(os.getenv("TOP_K", "5")),
            use_reranker=_env_bool("USE_RERANKER", False),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
            chunks_path=_resolve_path(os.getenv("CHUNKS_PATH", "data/chunks.jsonl"), project_root),
            chroma_dir=_resolve_path(os.getenv("CHROMA_DIR", "data/chroma"), project_root),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "regulasi_uu").strip(),
            google_api_key=api_key or None,
        )


def get_settings() -> Settings:
    """Muat file .env dari akar project lalu kembalikan objek Settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings.from_env()
