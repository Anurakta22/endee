"""Tests for the embedding module."""

import pytest
from unittest.mock import patch, MagicMock


def test_embed_returns_list_of_floats():
    mock_vector = [0.1] * 384
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: mock_vector)

    with patch("src.embedder._load_model", return_value=mock_model):
        from src.embedder import embed
        result = embed("Hello, world")
        assert isinstance(result, list)
        assert len(result) == 384


def test_embed_batch_returns_multiple_vectors():
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((3, 384))

    with patch("src.embedder._load_model", return_value=mock_model):
        from src.embedder import embed_batch
        results = embed_batch(["text one", "text two", "text three"])
        assert len(results) == 3


def test_embed_batch_empty():
    mock_model = MagicMock()
    mock_model.encode.return_value = []

    with patch("src.embedder._load_model", return_value=mock_model):
        from src.embedder import embed_batch
        results = embed_batch([])
        assert results == []
