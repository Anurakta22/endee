"""
Tests for MemoryStore – the Endee integration layer.

We mock the `endee` SDK so no live Endee server is needed.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.memory_store import MemoryEntry, MemoryStore


# ── MemoryEntry tests ─────────────────────────────────────────────────────────

def test_memory_entry_auto_id():
    entry = MemoryEntry(summary="User likes Python.", session_id="s1")
    assert entry.memory_id.startswith("s1_")
    assert entry.summary == "User likes Python."


def test_memory_entry_custom_id():
    entry = MemoryEntry(summary="Test", session_id="s1", memory_id="custom_id")
    assert entry.memory_id == "custom_id"


def test_memory_entry_to_vector_item_shape():
    with patch("src.memory_store.embed", return_value=[0.0] * 384):
        entry = MemoryEntry(
            summary="User prefers dark mode.",
            session_id="s1",
            role="user",
            turn=3,
            tags=["preference", "ui"],
        )
        item = entry.to_vector_item()

    assert "id" in item
    assert "vector" in item
    assert "meta" in item
    assert item["meta"]["summary"] == "User prefers dark mode."
    assert item["meta"]["tags"] == ["preference", "ui"]
    assert len(item["vector"]) == 384


def test_memory_entry_from_query_result():
    mock_result = MagicMock()
    mock_result.id = "s1_123"
    mock_result.meta = {
        "session_id": "s1",
        "summary": "Test summary",
        "role": "user",
        "turn": 5,
        "tags": ["a", "b"],
        "timestamp": "2026-01-01T00:00:00Z",
    }
    entry = MemoryEntry.from_query_result(mock_result)
    assert entry.session_id == "s1"
    assert entry.summary == "Test summary"
    assert entry.turn == 5


# ── MemoryStore tests ─────────────────────────────────────────────────────────

def _make_store():
    """Return a MemoryStore with a fully mocked Endee client."""
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.get_index.return_value = mock_index

    with patch("src.memory_store.Endee", return_value=mock_client):
        store = MemoryStore()
    store._index = mock_index
    store._client = mock_client
    return store, mock_index, mock_client


def test_save_calls_upsert():
    store, mock_index, _ = _make_store()
    with patch("src.memory_store.embed", return_value=[0.0] * 384):
        entry = MemoryEntry(summary="Test memory", session_id="s1")
        store.save(entry)

    mock_index.upsert.assert_called_once()
    upserted = mock_index.upsert.call_args[0][0]
    assert len(upserted) == 1
    assert upserted[0]["meta"]["summary"] == "Test memory"


def test_save_batch_calls_upsert_once():
    store, mock_index, _ = _make_store()
    with patch("src.memory_store.embed", return_value=[0.0] * 384):
        entries = [
            MemoryEntry(summary=f"Memory {i}", session_id="s1") for i in range(3)
        ]
        store.save_batch(entries)

    # Should be one upsert call with 3 items
    mock_index.upsert.assert_called_once()
    upserted = mock_index.upsert.call_args[0][0]
    assert len(upserted) == 3


def test_recall_returns_memory_entries():
    store, mock_index, _ = _make_store()

    # Build mock query results
    mock_results = []
    for i in range(3):
        r = MagicMock()
        r.id = f"s1_{i}"
        r.similarity = 0.9 - i * 0.1
        r.meta = {
            "session_id": "s1",
            "summary": f"Summary {i}",
            "role": "user",
            "turn": i,
            "tags": [],
            "timestamp": "2026-01-01T00:00:00Z",
        }
        mock_results.append(r)

    mock_index.query.return_value = mock_results

    with patch("src.memory_store.embed", return_value=[0.0] * 384):
        memories = store.recall("What did the user prefer?", top_k=3)

    assert len(memories) == 3
    assert memories[0].summary == "Summary 0"
    mock_index.query.assert_called_once()


def test_recall_filters_by_session():
    store, mock_index, _ = _make_store()

    mock_results = []
    for sess in ["s1", "s2", "s1"]:
        r = MagicMock()
        r.id = f"{sess}_0"
        r.similarity = 0.9
        r.meta = {
            "session_id": sess,
            "summary": f"Summary from {sess}",
            "role": "user",
            "turn": 1,
            "tags": [],
            "timestamp": "2026-01-01T00:00:00Z",
        }
        mock_results.append(r)

    mock_index.query.return_value = mock_results

    with patch("src.memory_store.embed", return_value=[0.0] * 384):
        memories = store.recall("query", top_k=5, session_id="s1")

    # Only s1 memories should be returned
    assert all(m.session_id == "s1" for m in memories)
    assert len(memories) == 2
