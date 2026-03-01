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

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "mistral")
    )
    mistral_api_key: str = field(
        default_factory=lambda: os.getenv("MISTRAL_API_KEY", "")
    )
    mistral_model: str = field(
        default_factory=lambda: os.getenv("MISTRAL_MODEL", "mistral-large-latest")
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
