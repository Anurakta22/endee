"""
MemoryStore
===========
The heart of AgentMemory.

Wraps the official `endee` Python SDK to provide a clean, typed interface for:
  - Initialising (and auto-creating) the Endee index
  - Storing episodic memory entries (upsert)
  - Retrieving the most semantically relevant memories (query)
  - Listing / deleting memories
  - Persisting across agent restarts  ← this is the key differentiator

HOW ENDEE IS USED
─────────────────
We create ONE index called `agent_memory` (configurable) with:
  • dimension  = 384  (all-MiniLM-L6-v2 output size)
  • space_type = "cosine"   (normalised dot-product → cosine similarity)
  • precision  = INT8   (quantised for speed/memory efficiency)

Each memory is stored as a vector item:
  {
    "id":     "<session_id>_<timestamp_ms>",
    "vector": [...384 floats...],   ← embedding of the summary text
    "meta": {
        "session_id":  "sess_abc",
        "summary":     "User prefers dark mode; asked about React hooks",
        "role":        "assistant",   # or "user"
        "timestamp":   "2026-02-27T10:30:00Z",
        "turn":        42,
        "tags":        ["preference", "react"]
    }
  }

At retrieval time we embed the current user message and call index.query()
to get the top-K cosine-nearest memories, then inject them into the LLM
system prompt as grounded long-term context.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from endee import Endee, Precision

from .config import cfg
from .embedder import embed


# ── Data model ────────────────────────────────────────────────────────────────

class MemoryEntry:
    """A single episodic memory record."""

    def __init__(
        self,
        summary: str,
        session_id: str,
        role: str = "assistant",
        turn: int = 0,
        tags: Optional[List[str]] = None,
        memory_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self.memory_id = memory_id or f"{session_id}_{int(time.time() * 1000)}_{uuid4().hex[:6]}"
        self.summary = summary
        self.session_id = session_id
        self.role = role
        self.turn = turn
        self.tags = tags or []
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_vector_item(self) -> Dict[str, Any]:
        """Serialise into the shape expected by Endee's index.upsert()."""
        return {
            "id": self.memory_id,
            "vector": embed(self.summary),
            "meta": {
                "session_id": self.session_id,
                "summary": self.summary,
                "role": self.role,
                "turn": self.turn,
                "tags": self.tags,
                "timestamp": self.timestamp,
            },
        }

    @classmethod
    def from_query_result(cls, result: Any) -> "MemoryEntry":
        """Reconstruct a MemoryEntry from an Endee query result object."""
        if isinstance(result, dict):
            meta = result.get("meta", {})
            return cls(
                memory_id=result.get("id"),
                summary=meta.get("summary", ""),
                session_id=meta.get("session_id", ""),
                role=meta.get("role", ""),
                turn=meta.get("turn", 0),
                tags=meta.get("tags", []),
                timestamp=meta.get("timestamp", ""),
            )
        else:
            meta = getattr(result, "meta", {}) or {}
            return cls(
                memory_id=getattr(result, "id", None),
                summary=meta.get("summary", ""),
                session_id=meta.get("session_id", ""),
                role=meta.get("role", ""),
                turn=meta.get("turn", 0),
                tags=meta.get("tags", []),
                timestamp=meta.get("timestamp", ""),
            )


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Persistent vector memory backed by Endee.

    Usage
    -----
    store = MemoryStore()          # connects & creates index if needed
    store.save(entry)              # upsert a memory
    memories = store.recall(query_text, session_id=None, top_k=5)
    """

    def __init__(self) -> None:
        # Initialise the official Endee Python SDK client
        self._client = Endee(cfg.endee_auth_token or None)
        self._client.set_base_url(f"{cfg.endee_base_url}/api/v1")
        self._index_name = cfg.endee_index_name
        self._index = self._get_or_create_index()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_or_create_index(self):
        """Return the Endee index, creating it on first run."""
        try:
            return self._client.get_index(self._index_name)
        except Exception:
            # Index does not exist yet → create it
            # In endee v0.1.16 this takes dimension, space_type, and precision at minimum
            self._client.create_index(
                name=self._index_name,
                dimension=cfg.embed_dimension,      # 384
                space_type="cosine",
                precision="float32",            # matches embedding raw output
            )
            return self._client.get_index(self._index_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, entry: MemoryEntry) -> None:
        """
        Persist a MemoryEntry into Endee.

        Internally calls index.upsert() with the embedded summary vector
        and full metadata payload.
        """
        self._index.upsert([entry.to_vector_item()])

    def save_batch(self, entries: List[MemoryEntry]) -> None:
        """Upsert multiple memories in a single API round-trip."""
        self._index.upsert([e.to_vector_item() for e in entries])

    def recall(
        self,
        query_text: str,
        top_k: int = None,
        session_id: Optional[str] = None,
        min_similarity: float = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve the top-K memories most semantically similar to query_text.

        Parameters
        ----------
        query_text    : The current user message / topic to match against.
        top_k         : Number of results (defaults to cfg.memory_top_k).
        session_id    : If set, restrict results to this session's memories.
        min_similarity: Drop results below this cosine similarity score.

        Returns
        -------
        List[MemoryEntry] sorted by descending relevance.
        """
        top_k = top_k or cfg.memory_top_k
        min_similarity = min_similarity if min_similarity is not None else cfg.importance_threshold

        query_vector = embed(query_text)

        # Endee query – returns list of result objects with .id, .similarity, .meta
        results = self._index.query(vector=query_vector, top_k=top_k)

        memories: List[MemoryEntry] = []
        for r in results:
            # Apply optional similarity threshold
            sim = r.get("similarity") if isinstance(r, dict) else getattr(r, "similarity", None)
            if sim is not None and sim < min_similarity:
                continue
            entry = MemoryEntry.from_query_result(r)
            # Apply optional session filter (client-side since Endee OSS
            # does not yet support metadata filters)
            if session_id and entry.session_id != session_id:
                continue
            memories.append(entry)

        return memories

    def recall_by_session(self, session_id: str, top_k: int = 20) -> List[MemoryEntry]:
        """
        Retrieve the most recent memories for a specific session.

        We use a neutral probe vector (zero-ish) and filter client-side.
        For deterministic ordering we sort by .turn descending.
        """
        # Use a short neutral query to get a broad result set
        results = self._index.query(
            vector=embed(f"session {session_id}"),
            top_k=top_k * 3,  # over-fetch then filter
        )
        memories = []
        for r in results:
            meta = r.get("meta", {}) if isinstance(r, dict) else (getattr(r, "meta", None) or {})
            if meta.get("session_id") == session_id:
                memories.append(MemoryEntry.from_query_result(r))
        memories.sort(key=lambda m: m.turn, reverse=True)
        return memories[:top_k]

    def stats(self) -> Dict[str, Any]:
        """Return basic index statistics from Endee."""
        try:
            idx_info = self._client.get_index(self._index_name)
            return {
                "index_name": self._index_name,
                "info": str(idx_info),
            }
        except Exception as exc:
            return {"error": str(exc)}
