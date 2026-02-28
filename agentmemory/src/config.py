"""Centralised configuration loaded from environment / .env file."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Endee ──────────────────────────────────────────────────────
    endee_base_url: str = field(
        default_factory=lambda: os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
    )
    endee_auth_token: str = field(
        default_factory=lambda: os.getenv("ENDEE_AUTH_TOKEN", "")
    )
    endee_index_name: str = field(
        default_factory=lambda: os.getenv("ENDEE_INDEX_NAME", "agent_memory")
    )

    # ── LLM ────────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3")
    )

    # ── Embeddings ─────────────────────────────────────────────────
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embed_dimension: int = 384  # dimension for all-MiniLM-L6-v2

    # ── Memory behaviour ───────────────────────────────────────────
    memory_top_k: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_TOP_K", "5"))
    )
    summary_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SUMMARY_MAX_TOKENS", "150"))
    )
    importance_threshold: float = field(
        default_factory=lambda: float(os.getenv("IMPORTANCE_THRESHOLD", "0.0"))
    )
    session_window: int = field(
        default_factory=lambda: int(os.getenv("SESSION_WINDOW", "20"))
    )


# Singleton used everywhere
cfg = Config()
