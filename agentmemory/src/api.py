"""
AgentMemory REST API
====================
A lightweight FastAPI server that exposes the MemoryAgent over HTTP.
Useful for integrating AgentMemory into larger systems or building a
frontend on top of it.

Endpoints
---------
POST /sessions                   → create / resume a session
POST /sessions/{session_id}/chat → send a message, get a reply
GET  /sessions/{session_id}/memories → list memories for a session
GET  /memories/search?q=...      → semantic search across all memories
GET  /health                     → liveness check
GET  /stats                      → Endee index stats
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import MemoryAgent
from .memory_store import MemoryStore

app = FastAPI(
    title="AgentMemory API",
    description="Long-term episodic memory for AI agents, powered by Endee vector DB.",
    version="1.0.0",
)

# In-memory session registry (sessions persist across requests within one process)
_sessions: dict[str, MemoryAgent] = {}


def _get_or_create(session_id: str) -> MemoryAgent:
    if session_id not in _sessions:
        _sessions[session_id] = MemoryAgent(session_id=session_id)
    return _sessions[session_id]


# ── Request / Response models ─────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    turn: int
    memories_used: int


class MemoryOut(BaseModel):
    memory_id: str
    summary: str
    session_id: str
    turn: int
    tags: List[str]
    timestamp: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    store = MemoryStore()
    return store.stats()


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(body: CreateSessionRequest):
    agent = _get_or_create(body.session_id or f"sess_{__import__('uuid').uuid4().hex[:8]}")
    return CreateSessionResponse(
        session_id=agent.session_id,
        message=f"Session '{agent.session_id}' ready.",
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, body: ChatRequest):
    agent = _get_or_create(session_id)

    # Recall memories before chat so we can report the count
    memories = agent.store.recall(body.message)

    answer = agent.chat(body.message)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        turn=agent._turn,
        memories_used=len(memories),
    )


@app.get("/sessions/{session_id}/memories", response_model=List[MemoryOut])
def get_session_memories(session_id: str):
    agent = _get_or_create(session_id)
    memories = agent.session_history()
    return [
        MemoryOut(
            memory_id=m.memory_id,
            summary=m.summary,
            session_id=m.session_id,
            turn=m.turn,
            tags=m.tags,
            timestamp=m.timestamp,
        )
        for m in memories
    ]


@app.post("/memories/search", response_model=List[MemoryOut])
def search_memories(body: SearchRequest):
    store = MemoryStore()
    memories = store.recall(query_text=body.query, top_k=body.top_k)
    return [
        MemoryOut(
            memory_id=m.memory_id,
            summary=m.summary,
            session_id=m.session_id,
            turn=m.turn,
            tags=m.tags,
            timestamp=m.timestamp,
        )
        for m in memories
    ]


@app.post("/sessions/{session_id}/checkpoint")
def checkpoint(session_id: str):
    agent = _get_or_create(session_id)
    saved = agent.force_checkpoint()
    return {"saved": saved}
