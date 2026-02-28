"""Tests for the FastAPI endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Return a TestClient with all dependencies mocked."""
    mock_store = MagicMock()
    mock_store.recall.return_value = []
    mock_store.recall_by_session.return_value = []
    mock_store.stats.return_value = {"index_name": "agent_memory"}

    with (
        patch("src.agent.MemoryStore", return_value=mock_store),
        patch("src.api.MemoryStore", return_value=mock_store),
        patch("src.agent.generate_answer", return_value="Test answer"),
        patch("src.agent.summarise_window", return_value="Summary"),
        patch("src.agent.extract_tags", return_value=[]),
    ):
        from src.api import app
        # Clear session registry between tests
        from src import api as api_module
        api_module._sessions.clear()
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_session(client):
    resp = client.post("/sessions", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"].startswith("sess_")


def test_create_session_with_custom_id(client):
    resp = client.post("/sessions", json={"session_id": "my_custom_sess"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my_custom_sess"


def test_chat_endpoint(client):
    # First create the session
    client.post("/sessions", json={"session_id": "chat_test"})

    with patch("src.agent.generate_answer", return_value="Great question!"):
        resp = client.post(
            "/sessions/chat_test/chat",
            json={"message": "What is Python?"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["session_id"] == "chat_test"
    assert data["turn"] >= 1


def test_get_session_memories(client):
    client.post("/sessions", json={"session_id": "mem_test"})
    resp = client.get("/sessions/mem_test/memories")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_search_memories(client):
    resp = client.post(
        "/memories/search",
        json={"query": "Python preferences", "top_k": 3},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
