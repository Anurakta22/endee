"""
Integration-level tests for MemoryAgent.

All LLM and Endee calls are mocked so tests run offline.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agent import MemoryAgent
from src.memory_store import MemoryEntry


def _make_agent(session_id="test_sess") -> tuple[MemoryAgent, MagicMock]:
    """Return an agent with mocked store and summariser."""
    mock_store = MagicMock()
    mock_store.recall.return_value = []
    mock_store.recall_by_session.return_value = []

    with (
        patch("src.agent.MemoryStore", return_value=mock_store),
        patch("src.agent.generate_answer", return_value="Mocked answer."),
        patch("src.agent.summarise_window", return_value="Mocked summary."),
        patch("src.agent.extract_tags", return_value=["test"]),
    ):
        agent = MemoryAgent(session_id=session_id)
        agent.store = mock_store

    return agent, mock_store


def test_chat_returns_answer():
    agent, mock_store = _make_agent()
    with (
        patch("src.agent.generate_answer", return_value="Hello back!"),
    ):
        answer = agent.chat("Hello")
    assert answer == "Hello back!"


def test_chat_increments_turn():
    agent, _ = _make_agent()
    with patch("src.agent.generate_answer", return_value="A"):
        agent.chat("msg1")
        agent.chat("msg2")
    assert agent._turn == 2


def test_chat_appends_to_buffer():
    agent, _ = _make_agent()
    with patch("src.agent.generate_answer", return_value="Reply"):
        agent.chat("User message")
    # Buffer should have 2 entries: user + assistant
    assert len(agent._buffer) == 2
    assert agent._buffer[0] == ("user", "User message")
    assert agent._buffer[1] == ("assistant", "Reply")


def test_checkpoint_triggered_on_full_buffer(monkeypatch):
    """When buffer reaches SESSION_WINDOW*2, _checkpoint should be called."""
    from src import config
    monkeypatch.setattr(config.cfg, "session_window", 2)  # window of 2 turns

    agent, mock_store = _make_agent()
    checkpoint_called = []

    original_checkpoint = agent._checkpoint
    def spy_checkpoint():
        checkpoint_called.append(True)
        original_checkpoint()

    agent._checkpoint = spy_checkpoint

    with (
        patch("src.agent.generate_answer", return_value="Reply"),
        patch("src.agent.summarise_window", return_value="Summary"),
        patch("src.agent.extract_tags", return_value=[]),
    ):
        # session_window=2 → buffer fills at 4 entries (2 turns × 2 msgs)
        agent.chat("msg1")
        agent.chat("msg2")

    assert checkpoint_called, "Checkpoint was not triggered"


def test_force_checkpoint_saves_to_store():
    agent, mock_store = _make_agent()
    agent._buffer = [("user", "hi"), ("assistant", "hello")]

    with (
        patch("src.agent.summarise_window", return_value="Summary text"),
        patch("src.agent.extract_tags", return_value=["greeting"]),
    ):
        result = agent.force_checkpoint()

    assert result is True
    mock_store.save.assert_called_once()
    saved_entry = mock_store.save.call_args[0][0]
    assert isinstance(saved_entry, MemoryEntry)
    assert saved_entry.summary == "Summary text"
    assert "greeting" in saved_entry.tags


def test_force_checkpoint_empty_buffer_returns_false():
    agent, mock_store = _make_agent()
    agent._buffer = []
    result = agent.force_checkpoint()
    assert result is False
    mock_store.save.assert_not_called()


def test_recall_delegates_to_store():
    agent, mock_store = _make_agent()
    dummy_memories = [MagicMock()]
    mock_store.recall.return_value = dummy_memories

    result = agent.recall("Python preferences")
    assert result == dummy_memories
    mock_store.recall.assert_called_with(query_text="Python preferences", top_k=5)


def test_session_history_delegates_to_store():
    agent, mock_store = _make_agent()
    agent.session_history()
    mock_store.recall_by_session.assert_called_with(agent.session_id)


def test_repr_contains_session_id():
    agent, _ = _make_agent(session_id="my_session")
    assert "my_session" in repr(agent)
