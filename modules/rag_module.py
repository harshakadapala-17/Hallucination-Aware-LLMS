"""
RAG Module
==========

Retrieval-Augmented Generation module using FAISS for vector search and
sentence-transformers for embeddings.

Advanced retrieval features:
  - HyDE (Hypothetical Document Embeddings) via Ollama
  - Multi-hop retrieval: two-stage retrieval using extracted key concepts
  - Cross-encoder re-ranking for higher precision
  - Query decomposition: splits complex queries into sub-questions
  - Contextual compression: keeps only query-relevant sentences per doc
  - Adaptive top-k: adjusts result count based on query complexity
  - Retrieval confidence scoring with direct-LLM fallback

All features are config-driven and degrade gracefully when dependencies
are unavailable.

**Inputs**:  Query string or list of documents (for index building).
**Outputs**: List of ``{ text, source, score }`` dicts.
**Dependencies**: faiss-cpu, sentence-transformers, numpy, PyYAML, ollama.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from modules import load_config

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_faiss = None
_SentenceTransformer = None
_cross_encoder_model = None
_cross_encoder_model_name = None  # track which model is loaded


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


def _get_cross_encoder(model_name: str):
    """Lazy-load the cross-encoder model for re-ranking.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        CrossEncoder instance, or None if unavailable.
    """
    global _cross_encoder_model, _cross_encoder_model_name
    if _cross_encoder_model is None or _cross_encoder_model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder(model_name)
            _cross_encoder_model_name = model_name
        except Exception:
            _cross_encoder_model = None
    return _cross_encoder_model


# ---------------------------------------------------------------------------
# RAGModule
# ---------------------------------------------------------------------------

class RAGModule:
    """Retrieve relevant documents for a query using FAISS + embeddings.

    Public API:
        retrieve(query, top_k)            — standard vector search
        retrieve_multihop(query, top_k)   — two-hop retrieval via entity concepts
        retrieve_decomposed(query, top_k) — sub-question decomposition retrieval
        rerank(query, documents)          — cross-encoder re-ranking
        compress_context(query, docs)     — sentence-level contextual compression
        get_adaptive_topk(complexity)     — complexity-driven top_k selection
        get_retrieval_confidence(docs)    — 0-1 confidence from top-doc scores
        build_index(documents)            — build and persist FAISS index
    """

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
        self.use_hyde: bool = bool(rag_cfg.get("use_hyde", False))

        self._model = None  # lazy-loaded SentenceTransformer
        self._index = None  # FAISS index
        self._documents: List[Dict[str, str]] = []  # source texts
        self._metadata_path = os.path.join(self.index_path, "metadata.jsonl")

        # Temporary state written during multihop / decomposed retrieval
        # (read by pipeline to populate the trace dict)
        self._last_followup_query: str = ""
        self._last_sub_questions: List[str] = []

        # Attempt to load existing index
        self._try_load_index()

    # ------------------------------------------------------------------ #
    #  Public API — standard retrieval                                    #
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

    # ------------------------------------------------------------------ #
    #  Public API — adaptive top-k                                        #
    # ------------------------------------------------------------------ #

    def get_adaptive_topk(self, complexity_score: float) -> int:
        """Return top_k tuned to query complexity.

        Low complexity  (< 0.3)  → small top_k (less noise).
        High complexity (> 0.7)  → large top_k (more context).

        Args:
            complexity_score: Float 0-1 from QueryAnalyzer.

        Returns:
            Integer top_k value configured in rag.adaptive_topk.
        """
        rag_cfg = self.config.get("rag", {})
        if not rag_cfg.get("use_adaptive_topk", True):
            return self.top_k

        adaptive_cfg = rag_cfg.get("adaptive_topk", {})
        low = int(adaptive_cfg.get("low", 2))
        medium = int(adaptive_cfg.get("medium", 3))
        high = int(adaptive_cfg.get("high", 5))

        if complexity_score < 0.3:
            return low
        elif complexity_score <= 0.7:
            return medium
        else:
            return high

    # ------------------------------------------------------------------ #
    #  Public API — multi-hop retrieval                                   #
    # ------------------------------------------------------------------ #

    def retrieve_multihop(
        self, query: str, top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval using key concepts from initial results.

        Step 1: Retrieve docs with the original query.
        Step 2: Extract key entities/concepts from those docs.
        Step 3: Form a follow-up query = original + extracted concepts.
        Step 4: Retrieve docs with the follow-up query.
        Step 5: Merge, deduplicate by text prefix, re-rank by score.

        Side effect: stores the follow-up query in ``self._last_followup_query``
        so the pipeline can include it in the trace.

        Args:
            query: User query string.
            top_k: Number of final results to return.

        Returns:
            Merged top_k document list.
        """
        if top_k is None:
            top_k = self.top_k

        # Step 1 — initial retrieval
        first_results = self.retrieve(query, top_k=top_k)
        if not first_results:
            self._last_followup_query = query
            return []

        # Step 2 — extract concepts from top-2 docs
        combined_text = " ".join(doc["text"] for doc in first_results[:2])
        entities = self._extract_key_concepts(combined_text)

        # Step 3 — form follow-up query
        if entities:
            follow_up = f"{query} {' '.join(entities[:3])}"
        else:
            follow_up = query
        self._last_followup_query = follow_up

        # Step 4 — second retrieval
        second_results = self.retrieve(follow_up, top_k=top_k)

        # Step 5 — merge, deduplicate, sort by score
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for doc in first_results + second_results:
            key = doc["text"][:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return merged[:top_k]

    # ------------------------------------------------------------------ #
    #  Public API — query decomposition retrieval                         #
    # ------------------------------------------------------------------ #

    def decompose_query(self, query: str) -> List[str]:
        """Break a complex query into 2-3 simpler sub-questions.

        Tries Ollama first; falls back to heuristic conjunction splitting.

        Args:
            query: Complex user query string.

        Returns:
            List of 2-3 simpler sub-question strings.
        """
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "llama3.2")

        try:
            import ollama  # noqa: F401 (checked at runtime)
            prompt = (
                "Break this complex question into 2-3 simple sub-questions that "
                "together fully answer the original. "
                "Return ONLY a JSON array of strings, no explanation, no markdown.\n\n"
                f"Question: {query}"
            )
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"].strip()
            # Extract first JSON array from the response
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                sub_qs = json.loads(match.group())
                if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                    return [str(q) for q in sub_qs[:3]]
        except Exception:
            pass

        return self._heuristic_decompose(query)

    def retrieve_decomposed(
        self, query: str, top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for each sub-question and merge results.

        Step 1: Decompose query into 2-3 sub-questions.
        Step 2: Retrieve top_k // n_sub_questions docs per sub-question.
        Step 3: Merge, deduplicate, sort by score.

        Side effect: stores sub-questions in ``self._last_sub_questions``.

        Args:
            query: Complex user query string.
            top_k: Total number of results to return.

        Returns:
            Merged top_k document list.
        """
        if top_k is None:
            top_k = self.top_k

        sub_questions = self.decompose_query(query)
        self._last_sub_questions = sub_questions

        if not sub_questions:
            return self.retrieve(query, top_k=top_k)

        per_q_k = max(1, top_k // len(sub_questions))

        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for sub_q in sub_questions:
            results = self.retrieve(sub_q, top_k=per_q_k)
            for doc in results:
                key = doc["text"][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return merged[:top_k]

    # ------------------------------------------------------------------ #
    #  Public API — cross-encoder re-ranking                              #
    # ------------------------------------------------------------------ #

    def rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank documents using a cross-encoder relevance model.

        Each (query, doc_text) pair is scored; documents are sorted by
        that score descending.  Falls back to original order gracefully
        if the cross-encoder is unavailable.

        Args:
            query: User query string.
            documents: List of document dicts (must have ``text`` key).

        Returns:
            Documents sorted by cross-encoder score descending.
            Each dict gains a ``rerank_score`` field.
        """
        if not documents:
            return []

        rag_cfg = self.config.get("rag", {})
        model_name = rag_cfg.get(
            "reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        try:
            ce = _get_cross_encoder(model_name)
            if ce is None:
                return documents
            pairs = [(query, doc["text"]) for doc in documents]
            scores = ce.predict(pairs)
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            documents.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        except Exception:
            pass  # return original order

        return documents

    # ------------------------------------------------------------------ #
    #  Public API — contextual compression                                #
    # ------------------------------------------------------------------ #

    def compress_context(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract only query-relevant sentences from each document.

        Sentences with cosine similarity to the query embedding above
        ``rag.compression_threshold`` are kept.  If none qualify, the
        top-2 by similarity are kept anyway to avoid empty context.

        Args:
            query: User query for relevance scoring.
            documents: Retrieved document dicts.

        Returns:
            New list of dicts where ``text`` contains only relevant
            sentences; each dict gains ``compressed: True``.
        """
        if not documents:
            return []

        rag_cfg = self.config.get("rag", {})
        threshold = float(rag_cfg.get("compression_threshold", 0.4))

        try:
            model = self._get_model()
            query_vec = model.encode([query], normalize_embeddings=True)[0]

            compressed: List[Dict[str, Any]] = []
            for doc in documents:
                text = doc.get("text", "")
                # Split on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", text.strip())
                sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

                if not sentences:
                    compressed.append(dict(doc))
                    continue

                sent_vecs = model.encode(sentences, normalize_embeddings=True)
                sims = np.dot(sent_vecs, query_vec)

                kept = [s for s, sim in zip(sentences, sims) if sim >= threshold]

                # Fallback: always keep at least 2 sentences
                if not kept:
                    top_idx = np.argsort(sims)[-2:][::-1]
                    kept = [sentences[i] for i in sorted(top_idx) if i < len(sentences)]

                new_doc = dict(doc)
                new_doc["text"] = " ".join(kept)
                new_doc["compressed"] = True
                compressed.append(new_doc)

            return compressed

        except Exception:
            return documents  # return uncompressed on failure

    # ------------------------------------------------------------------ #
    #  Public API — retrieval confidence                                  #
    # ------------------------------------------------------------------ #

    def get_retrieval_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Compute retrieval confidence from top-3 document scores.

        Uses ``rerank_score`` when available (from cross-encoder), otherwise
        the raw FAISS inner-product ``score``.  Cross-encoder scores are
        sigmoid-normalised to [0, 1].

        Args:
            documents: Retrieved (and optionally re-ranked) document list.

        Returns:
            Float in [0, 1].  0.0 when documents is empty.
        """
        if not documents:
            return 0.0

        has_rerank = any("rerank_score" in d for d in documents[:3])
        scores: List[float] = []

        for doc in documents[:3]:
            if has_rerank:
                raw = doc.get("rerank_score", doc.get("score", 0.0))
                # Sigmoid-normalise cross-encoder logits → (0, 1)
                score = 1.0 / (1.0 + math.exp(-raw))
            else:
                # FAISS inner-product on unit vecs is already ∈ [-1, 1]
                # shift to [0, 1]
                raw = doc.get("score", 0.0)
                score = (raw + 1.0) / 2.0
            scores.append(score)

        confidence = sum(scores) / len(scores) if scores else 0.0
        return round(min(1.0, max(0.0, confidence)), 4)

    # ------------------------------------------------------------------ #
    #  Public API — index building                                        #
    # ------------------------------------------------------------------ #

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

        chunks = self._chunk_documents(documents)
        self._documents = chunks

        model = self._get_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        faiss = _get_faiss()
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._save_index()

    # ------------------------------------------------------------------ #
    #  Agentic query reformulation (Ollama)                              #
    # ------------------------------------------------------------------ #

    def reformulate_query(
        self,
        original_query: str,
        context: str,
        reason: str,
    ) -> str:
        """Reformulate a query based on why the previous attempt failed.

        Used by the agent loop in pipeline.py to produce a more specific or
        more retrievable version of the query after a failed iteration.

        Args:
            original_query: The original user question.
            context: A short string describing what happened in the previous
                attempt (e.g. the partial answer or "No relevant documents found").
            reason: One of "low_retrieval_confidence", "refuted",
                "partially_supported".

        Returns:
            Reformulated query string, or the original query unchanged if
            Ollama is unavailable or the call fails.
        """
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "llama3.2")

        reason_hints = {
            "low_retrieval_confidence": (
                "no relevant documents were found in the knowledge base"
            ),
            "refuted": (
                "the generated answer contained claims that could not be verified "
                "and were flagged as potentially incorrect"
            ),
            "partially_supported": (
                "the generated answer was only partially supported by the "
                "retrieved documents"
            ),
        }
        reason_text = reason_hints.get(reason, reason)

        prompt = (
            f"Given this original question: {original_query}\n"
            f"The previous attempt failed because: {reason_text}\n"
            f"Context from previous attempt: {context}\n"
            "Reformulate the question to be more specific and retrievable.\n"
            "Return ONLY the reformulated question, nothing else."
        )

        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            reformulated = response["message"]["content"].strip()
            # Guard: if Ollama returns empty or very long output, fall back
            if not reformulated or len(reformulated) > 500:
                return original_query
            return reformulated
        except Exception:
            return original_query

    # ------------------------------------------------------------------ #
    #  HyDE — Hypothetical Document Embeddings (Ollama)                  #
    # ------------------------------------------------------------------ #

    def _hypothetical_answer(self, query: str) -> str:
        """Generate a short hypothetical answer via Ollama for HyDE retrieval.

        The hypothetical answer is embedded and used as the retrieval query
        instead of the raw question, improving recall for factual queries.

        Args:
            query: The user's question.

        Returns:
            Short answer text, or empty string on failure.
        """
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "llama3.2")

        try:
            import ollama  # noqa: F401
            prompt = (
                "Write a short, factual, one-paragraph answer to the question below. "
                "Be concise (2-3 sentences). "
                "Your answer will be used only for document retrieval.\n\n"
                f"Question: {query}"
            )
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_key_concepts(text: str) -> List[str]:
        """Extract capitalised noun phrases as key concepts for multi-hop query.

        Args:
            text: Combined text from initial retrieval results.

        Returns:
            Up to 5 unique capitalised noun-phrase strings.
        """
        pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        matches = pattern.findall(text)
        stop = {
            "The", "A", "An", "This", "That", "These", "Those",
            "It", "He", "She", "They", "We", "You",
        }
        seen: set = set()
        result: List[str] = []
        for m in matches:
            if m not in stop and m not in seen:
                seen.add(m)
                result.append(m)
        return result[:5]

    @staticmethod
    def _heuristic_decompose(query: str) -> List[str]:
        """Heuristic query decomposition by splitting on conjunctions.

        Args:
            query: User query string.

        Returns:
            List of sub-question strings (at least 1, at most 3).
        """
        parts = re.split(
            r"\s+(?:and|as well as|both|compare|also|additionally|furthermore)\s+",
            query,
            flags=re.IGNORECASE,
        )
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts[:3]
        # No conjunction found — return the original as a single sub-question
        return [query]

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
        """Lazy-load the sentence transformer embedding model."""
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
        """Try to load a previously saved FAISS index from disk."""
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


# ---------------------------------------------------------------------------
#  Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rag = RAGModule()

    print("=== RAG Module (no index) ===")
    results = rag.retrieve("What is the capital of France?")
    print(f"  Results: {results}")
    print("  (empty is expected when no index has been built)")

    print("\n=== Building tiny index ===")
    docs = [
        {
            "text": "Paris is the capital and largest city of France, situated on the river Seine.",
            "source": "wiki_france",
        },
        {
            "text": "Berlin is the capital of Germany and one of the 16 states of Germany.",
            "source": "wiki_germany",
        },
        {
            "text": "Tokyo is the capital of Japan and the most populous city in the world.",
            "source": "wiki_japan",
        },
    ]
    rag.build_index(docs)
    print("  Index built successfully.")

    print("\n=== Standard retrieval ===")
    results = rag.retrieve("What is the capital of France?", top_k=2)
    for r in results:
        print(f"  score={r['score']:.4f}  source={r['source']}  text={r['text'][:60]}...")

    print("\n=== Adaptive top_k ===")
    for complexity in [0.1, 0.5, 0.9]:
        k = rag.get_adaptive_topk(complexity)
        print(f"  complexity={complexity} → top_k={k}")

    print("\n=== Retrieval confidence ===")
    conf = rag.get_retrieval_confidence(results)
    print(f"  Confidence: {conf:.4f}")
