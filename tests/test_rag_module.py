"""
Tests for modules.rag_module.RAGModule
=======================================

Covers:
  - Empty index returns empty list
  - Build index + retrieve happy path
  - Edge cases: empty query, non-string query, empty documents
  - Chunking logic
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from modules.rag_module import RAGModule


def _make_config(tmpdir: str) -> Dict[str, Any]:
    """Create a config with temp index path."""
    return {
        "rag": {
            "top_k": 2,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "index_path": str(Path(tmpdir) / "faiss_index"),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "model": {"name": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": 256},
    }


SAMPLE_DOCS = [
    {"text": "Paris is the capital of France.", "source": "doc1"},
    {"text": "Berlin is the capital of Germany.", "source": "doc2"},
    {"text": "Tokyo is the capital of Japan.", "source": "doc3"},
]


# ------------------------------------------------------------------ #
#  Happy-path                                                          #
# ------------------------------------------------------------------ #

class TestHappyPath:
    def test_no_index_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            results = rag.retrieve("hello")
            assert results == []

    def test_build_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            rag.build_index(SAMPLE_DOCS)

            results = rag.retrieve("capital of France")
            assert len(results) > 0
            assert all(
                {"text", "source", "score"} <= set(r.keys()) for r in results
            )
            # Top result should be France
            assert "France" in results[0]["text"] or "Paris" in results[0]["text"]

    def test_top_k_limits_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            rag.build_index(SAMPLE_DOCS)

            results = rag.retrieve("capital", top_k=1)
            assert len(results) == 1


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_non_string_query_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            with pytest.raises(TypeError):
                rag.retrieve(123)  # type: ignore[arg-type]

    def test_empty_documents_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            with pytest.raises(ValueError, match="empty"):
                rag.build_index([])

    def test_index_persists_to_disk(self) -> None:
        """Build index, create new RAGModule instance, verify it loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag1 = RAGModule(config=cfg)
            rag1.build_index(SAMPLE_DOCS)

            # New instance should auto-load
            rag2 = RAGModule(config=cfg)
            results = rag2.retrieve("France")
            assert len(results) > 0


# ------------------------------------------------------------------ #
#  Chunking                                                            #
# ------------------------------------------------------------------ #

class TestChunking:
    def test_short_doc_not_chunked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            rag = RAGModule(config=cfg)
            chunks = rag._chunk_documents([{"text": "Short text", "source": "s"}])
            assert len(chunks) == 1

    def test_long_doc_chunked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(tmpdir)
            cfg["rag"]["chunk_size"] = 50
            cfg["rag"]["chunk_overlap"] = 10
            rag = RAGModule(config=cfg)
            long_text = "A" * 200
            chunks = rag._chunk_documents([{"text": long_text, "source": "s"}])
            assert len(chunks) > 1
            # All chunks should have text
            assert all(len(c["text"]) > 0 for c in chunks)
