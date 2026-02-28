"""
MemoryAgent
===========
The top-level orchestrator that ties everything together:

  User message
      │
      ▼
  ① Embed the message → query Endee for relevant past memories
      │
      ▼
  ② Inject memories into LLM system prompt → generate answer
      │
      ▼
  ③ Append (user, assistant) turns to the in-session buffer
      │
      ▼
  ④ Every SESSION_WINDOW turns → summarise buffer → upsert to Endee
      │
      ▼
  Answer returned to caller

The agent also exposes `force_checkpoint()` to flush memory immediately
(useful on session end or SIGINT).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4

from .config import cfg
from .memory_store import MemoryEntry, MemoryStore
from .summariser import extract_tags, generate_answer, summarise_window


class MemoryAgent:
    """
    A stateful AI agent with cross-session episodic memory.

    Parameters
    ----------
    session_id : Unique identifier for this conversation session.
                 Re-using the same session_id lets you continue a session
                 across process restarts (memories are retrieved from Endee).
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or f"sess_{uuid4().hex[:8]}"
        self.store = MemoryStore()

        # In-session message buffer  [(role, content), ...]
        self._buffer: List[Tuple[str, str]] = []
        self._turn: int = 0

        # Track total memories saved this session
        self._memories_saved: int = 0

    # ── Core chat method ──────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process one user turn and return the assistant's response.

        Steps
        -----
        1. Recall semantically relevant past memories from Endee.
        2. Generate a contextualised answer.
        3. Buffer both turns.
        4. Checkpoint to Endee if the buffer is full.
        """
        self._turn += 1

        # ── Step 1: Recall ────────────────────────────────────────────────────
        relevant_memories = self.store.recall(
            query_text=user_message,
            top_k=cfg.memory_top_k,
        )

        # ── Step 2: Generate ──────────────────────────────────────────────────
        answer = generate_answer(
            user_message=user_message,
            memories=relevant_memories,
            chat_history=self._buffer,
        )

        # ── Step 3: Buffer ────────────────────────────────────────────────────
        self._buffer.append(("user", user_message))
        self._buffer.append(("assistant", answer))

        # ── Step 4: Checkpoint ────────────────────────────────────────────────
        if len(self._buffer) >= cfg.session_window * 2:
            self._checkpoint()

        return answer

    # ── Memory management ─────────────────────────────────────────────────────

    def _checkpoint(self) -> None:
        """
        Summarise the current buffer and persist it to Endee.
        Clears the buffer afterwards (keeps the last 2 turns for continuity).
        """
        if not self._buffer:
            return

        summary = summarise_window(self._buffer)
        if not summary:
            return

        tags = extract_tags(summary)

        entry = MemoryEntry(
            summary=summary,
            session_id=self.session_id,
            role="mixed",
            turn=self._turn,
            tags=tags,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.store.save(entry)
        self._memories_saved += 1

        # Retain the last exchange for context continuity
        self._buffer = self._buffer[-2:]

    def force_checkpoint(self) -> bool:
        """
        Manually flush the current buffer to Endee.
        Call this on session end or SIGINT to avoid losing recent context.

        Returns True if a memory was saved, False if buffer was empty.
        """
        if not self._buffer:
            return False
        self._checkpoint()
        return True

    def recall(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Expose the store's recall for direct inspection / debugging."""
        return self.store.recall(query_text=query, top_k=top_k)

    def session_history(self) -> List[MemoryEntry]:
        """Return all stored memories tagged with this session_id."""
        return self.store.recall_by_session(self.session_id)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MemoryAgent(session_id={self.session_id!r}, "
            f"turn={self._turn}, "
            f"buffer_len={len(self._buffer)}, "
            f"memories_saved={self._memories_saved})"
        )
