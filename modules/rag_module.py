"""
RAG Module
==========

Retrieval-Augmented Generation module using FAISS for vector search and
sentence-transformers for embeddings.

Supports HyDE (Hypothetical Document Embeddings): instead of embedding the
raw query, a short hypothetical answer is generated and its embedding is used
for retrieval. This dramatically improves recall for factual queries because
the search space aligns with answer-style text rather than question-style text.

HyDE is enabled by default (rag.use_hyde: true in config). When the OpenAI
key is absent or the generation fails, it falls back to direct query embedding.

**Inputs**:  Query string or list of documents (for index building).
**Outputs**: List of ``{ text, source, score }`` dicts.
**Dependencies**: faiss-cpu, sentence-transformers, numpy, PyYAML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modules import load_config

# Lazy imports to speed up module loading when not needed
_faiss = None
_SentenceTransformer = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class RAGModule:
    """Retrieve relevant documents for a query using FAISS + embeddings."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the RAGModule.

        Args:
            config: Pre-loaded configuration dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()
        rag_cfg = self.config.get("rag", {})

        self.top_k: int = int(rag_cfg.get("top_k", 3))
        self.chunk_size: int = int(rag_cfg.get("chunk_size", 200))
        self.chunk_overlap: int = int(rag_cfg.get("chunk_overlap", 50))
        self.index_path: str = rag_cfg.get("index_path", "data/faiss_index")
        self.embedding_model_name: str = rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.use_hyde: bool = bool(rag_cfg.get("use_hyde", True))

        self._model = None  # lazy-loaded SentenceTransformer
        self._index = None  # FAISS index
        self._documents: List[Dict[str, str]] = []  # source texts
        self._metadata_path = os.path.join(self.index_path, "metadata.jsonl")

        # Attempt to load existing index
        self._try_load_index()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Retrieve the most relevant documents for *query*.

        Args:
            query: User query string.
            top_k: Number of results to return (overrides config).

        Returns:
            List of dicts with ``text``, ``source``, and ``score`` keys,
            ordered by descending relevance.  Returns empty list when
            no index is loaded.

        Raises:
            TypeError: If *query* is not a str.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        if top_k is None:
            top_k = self.top_k

        if self._index is None or len(self._documents) == 0:
            return []

        model = self._get_model()

        # HyDE: embed a hypothetical answer instead of the raw query
        search_text = query
        if self.use_hyde:
            hyde_answer = self._hypothetical_answer(query)
            if hyde_answer:
                search_text = hyde_answer

        query_vec = model.encode([search_text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        faiss = _get_faiss()
        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            results.append({
                "text": doc.get("text", ""),
                "source": doc.get("source", "unknown"),
                "score": round(float(dist), 4),
            })

        return results

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build a FAISS index from a list of documents.

        Each document dict should have at least a ``text`` key and
        optionally a ``source`` key.

        Args:
            documents: List of ``{text: str, source: str}`` dicts.

        Raises:
            ValueError: If *documents* is empty.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        # Chunk documents
        chunks = self._chunk_documents(documents)
        self._documents = chunks

        # Encode
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index (Inner Product on normalised vecs ≈ cosine sim)
        faiss = _get_faiss()
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        # Persist
        self._save_index()

    # ------------------------------------------------------------------ #
    #  HyDE — Hypothetical Document Embeddings                           #
    # ------------------------------------------------------------------ #

    def _hypothetical_answer(self, query: str) -> str:
        """Generate a short hypothetical answer to use as the retrieval query.

        Uses the OpenAI API if available; returns empty string on failure so
        the caller falls back to direct query embedding.

        Args:
            query: The user's question.

        Returns:
            Short hypothetical answer text, or empty string on failure.
        """
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return ""

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            model_name = self.config.get("model", {}).get("name", "gpt-3.5-turbo")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a short, factual, one-paragraph answer to the "
                            "question below. Be concise (2-3 sentences max). "
                            "Your answer will be used only for document retrieval."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    #  Chunking                                                           #
    # ------------------------------------------------------------------ #

    def _chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Split documents into overlapping character-level chunks.

        Args:
            documents: Raw document list.

        Returns:
            List of chunk dicts with ``text`` and ``source``.
        """
        chunks: List[Dict[str, str]] = []
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            if len(text) <= self.chunk_size:
                chunks.append({"text": text, "source": source})
                continue

            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append({"text": chunk_text, "source": source})
                start += self.chunk_size - self.chunk_overlap

        return chunks

    # ------------------------------------------------------------------ #
    #  Model & persistence                                                #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        Path(self.index_path).mkdir(parents=True, exist_ok=True)

        faiss = _get_faiss()
        index_file = os.path.join(self.index_path, "index.faiss")
        faiss.write_index(self._index, index_file)

        with open(self._metadata_path, "w", encoding="utf-8") as f:
            for doc in self._documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    def _try_load_index(self) -> None:
        """Try to load a previously saved FAISS index."""
        index_file = os.path.join(self.index_path, "index.faiss")
        if not os.path.exists(index_file) or not os.path.exists(self._metadata_path):
            return

        try:
            faiss = _get_faiss()
            self._index = faiss.read_index(index_file)

            self._documents = []
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._documents.append(json.loads(line))
        except Exception:
            self._index = None
            self._documents = []


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    rag = RAGModule()

    print("=== RAG Module (no index) ===")
    results = rag.retrieve("What is the capital of France?")
    print(f"  Results: {results}")
    print("  (empty is expected when no index has been built)")

    # Build a tiny index for demo
    print("\n=== Building tiny index ===")
    docs = [
        {"text": "Paris is the capital and largest city of France, situated on the river Seine.", "source": "wiki_france"},
        {"text": "Berlin is the capital of Germany and one of the 16 states of Germany.", "source": "wiki_germany"},
        {"text": "Tokyo is the capital of Japan and the most populous city in the world.", "source": "wiki_japan"},
    ]
    rag.build_index(docs)
    print("  Index built successfully.")

    print("\n=== Retrieving ===")
    results = rag.retrieve("What is the capital of France?", top_k=2)
    for r in results:
        print(f"  score={r['score']:.4f}  source={r['source']}  text={r['text'][:60]}...")
