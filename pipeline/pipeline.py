"""
Pipeline Module
===============

Orchestrates the full Hallucination-Aware Adaptive LLM pipeline:
  QueryAnalyzer → HallucinationPredictor → StrategySelector →
  (RAGModule) → GenerationModule → (VerificationModule)

Advanced retrieval routing:
  - Adaptive top_k based on query complexity
  - Decomposed retrieval for citation-heavy or high-complexity queries
  - Multi-hop retrieval for multi-hop indicator queries
  - Cross-encoder re-ranking after retrieval
  - Retrieval confidence check with direct-LLM fallback
  - Contextual compression before generation

**Inputs**:  A raw query string.
**Outputs**: A full trace dict (see PIPELINE TRACE FORMAT in the PRD).
**Dependencies**: All six core modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules import load_config
from modules.query_analyzer import QueryAnalyzer
from modules.hallucination_predictor import HallucinationPredictor
from modules.strategy_selector import StrategySelector
from modules.generation_module import GenerationModule
from modules.rag_module import RAGModule
from modules.verification_module import VerificationModule


class Pipeline:
    """End-to-end hallucination-aware LLM pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise all pipeline components.

        Args:
            config: Pre-loaded config dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()

        self.query_analyzer = QueryAnalyzer(config=self.config)
        self.predictor = HallucinationPredictor(config=self.config)
        self.selector = StrategySelector(config=self.config)
        self.generator = GenerationModule(config=self.config)
        self.rag = RAGModule(config=self.config)
        self.verifier = VerificationModule(config=self.config)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, query: str) -> Dict[str, Any]:
        """Run the full pipeline on a single query.

        Steps:
          1.  Analyse query features.
          2.  Predict hallucination risk.
          3.  Select answering strategy.
          4.  Retrieve documents (adaptive top_k, routing by complexity).
          4b. Re-rank retrieved documents with cross-encoder.
          4c. Check retrieval confidence; fall back to direct_llm if low.
          4d. Compress context to query-relevant sentences.
          5.  Generate answer.
          6.  Verify answer (if strategy is rag_verification).
          6b. Re-generate if verdict is 'refuted'.
          7.  Return full trace dict.

        Args:
            query: The user's question.

        Returns:
            Trace dict with all pipeline fields including new retrieval
            intelligence fields (retrieval_strategy, reranking_applied,
            compression_applied, adaptive_topk_used, retrieval_confidence,
            retrieval_details).

        Raises:
            TypeError: If *query* is not a str.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        # Step 1: Analyse
        features: Dict[str, Any] = self.query_analyzer.analyze(query)

        # Step 2: Predict — pass query so self-consistency can fire when enabled
        prediction: Dict[str, Any] = self.predictor.predict(features, query=query)

        # Step 3: Select strategy (pass features to enable hard overrides)
        strategy: str = self.selector.select(prediction, features=features)

        # Determine whether the strategy was set by a hard override rule so the
        # confidence fallback can be suppressed for queries that genuinely need
        # grounded answers regardless of retrieval quality.
        _complexity = float(features.get("complexity_score", 0.0))
        hard_override: bool = bool(
            (features.get("multi_hop_indicator") and _complexity > 0.6)
            or features.get("contains_citation_pattern")
            or (_complexity > 0.75)
        )

        # ── Retrieval intelligence defaults ───────────────────────────────
        retrieved_docs: List[Dict[str, Any]] = []
        retrieval_strategy: str = "standard"
        adaptive_topk_used: int = self.config.get("rag", {}).get("top_k", 3)
        reranking_applied: bool = False
        compression_applied: bool = False
        retrieval_confidence: float = 0.0
        retrieval_details: Dict[str, Any] = {}

        # Step 4: Retrieve (only for rag / rag_verification)
        if strategy in ("rag", "rag_verification"):
            rag_cfg = self.config.get("rag", {})
            complexity = float(features.get("complexity_score", 0.5))

            # 4a — adaptive top_k
            if rag_cfg.get("use_adaptive_topk", True):
                top_k = self.rag.get_adaptive_topk(complexity)
            else:
                top_k = int(rag_cfg.get("top_k", 3))
            adaptive_topk_used = top_k

            # 4b — choose retrieval method
            multi_hop = bool(features.get("multi_hop_indicator", False))
            citation = bool(features.get("contains_citation_pattern", False))
            high_complexity = complexity > 0.7
            use_decomp = rag_cfg.get("use_query_decomposition", True)

            if use_decomp and (citation or high_complexity):
                # Decomposed retrieval: most thorough, covers high-complexity
                retrieved_docs = self.rag.retrieve_decomposed(query, top_k=top_k)
                retrieval_strategy = "decomposed"
                retrieval_details["sub_questions"] = list(
                    getattr(self.rag, "_last_sub_questions", [])
                )
            elif multi_hop:
                # Multi-hop: two-stage retrieval using extracted concepts
                retrieved_docs = self.rag.retrieve_multihop(query, top_k=top_k)
                retrieval_strategy = "multihop"
                retrieval_details["follow_up_query"] = getattr(
                    self.rag, "_last_followup_query", ""
                )
            else:
                # Standard vector search
                retrieved_docs = self.rag.retrieve(query, top_k=top_k)
                retrieval_strategy = "standard"

            # 4c — cross-encoder re-ranking
            if rag_cfg.get("use_reranking", True) and retrieved_docs:
                retrieved_docs = self.rag.rerank(query, retrieved_docs)
                reranking_applied = True

            # 4d — retrieval confidence check
            retrieval_confidence = self.rag.get_retrieval_confidence(retrieved_docs)
            min_conf = float(rag_cfg.get("min_retrieval_confidence", 0.3))

            if retrieved_docs and retrieval_confidence < min_conf and not hard_override:
                # Poor retrieval AND no hard override: fall back to direct LLM.
                # Hard-override queries (citation, multi-hop, high complexity) must
                # remain grounded even if the index is sparse — falling back to a
                # bare LLM would be worse than using low-confidence retrieved docs.
                strategy = "direct_llm"
                retrieved_docs = []
                retrieval_details["confidence_fallback"] = True
            elif retrieved_docs and retrieval_confidence < min_conf and hard_override:
                # Low confidence but hard override active — keep docs, flag it.
                retrieval_details["low_confidence_override"] = True
            if rag_cfg.get("use_contextual_compression", True) and retrieved_docs:
                # 4e — contextual compression
                retrieved_docs = self.rag.compress_context(query, retrieved_docs)
                compression_applied = True

        # Step 5: Generate
        context_texts = [doc["text"] for doc in retrieved_docs]
        generation: Dict[str, Any] = self.generator.generate(
            query, strategy, context=context_texts
        )

        # Step 6: Verify (only for rag_verification)
        verification: Optional[Dict[str, Any]] = None
        if strategy == "rag_verification" and retrieved_docs:
            verification = self.verifier.verify(
                generation["answer"], retrieved_docs
            )

            # Step 6b: Re-generate if answer was refuted
            max_retries = self.config.get("pipeline", {}).get("max_regeneration_retries", 1)
            retries = 0
            while (
                verification.get("verdict") == "refuted"
                and retries < max_retries
            ):
                generation = self.generator.generate(
                    query,
                    "rag_verification",
                    context=context_texts,
                    retry_hint=(
                        "Your previous answer contained claims that could not be "
                        "verified. Answer ONLY using the provided documents. "
                        "If the documents do not support an answer, say so explicitly."
                    ),
                )
                verification = self.verifier.verify(
                    generation["answer"], retrieved_docs
                )
                retries += 1

            if retries > 0:
                generation["regenerated"] = retries

        # Step 7: Assemble trace
        trace: Dict[str, Any] = {
            "query": query,
            "features": features,
            "prediction": prediction,
            "strategy": strategy,
            "retrieved_docs": retrieved_docs,
            "generation": generation,
            "verification": verification,
            # ── Retrieval intelligence ────────────────────────────────
            "retrieval_strategy": retrieval_strategy,
            "adaptive_topk_used": adaptive_topk_used,
            "reranking_applied": reranking_applied,
            "compression_applied": compression_applied,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_details": retrieval_details,
            # ── Hard override ─────────────────────────────────────────
            "hard_override_applied": hard_override,
            # ── Hybrid scoring ────────────────────────────────────────
            "hybrid_scoring": prediction.get("hybrid_scoring", False),
            "self_consistency_score": (
                prediction.get("self_consistency", {}).get("consistency_score")
                if prediction.get("hybrid_scoring") else None
            ),
            "feature_risk_score": prediction.get("feature_risk_score"),
        }

        return trace


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    pipeline = Pipeline()

    queries = [
        "What is the capital of France?",
        "According to Dr. Smith, what caused the 2008 crisis and how did it compare to the Great Depression?",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print("=" * 70)
        trace = pipeline.run(q)
        print(f"  Strategy:          {trace['strategy']}")
        print(f"  Retrieval:         {trace['retrieval_strategy']}")
        print(f"  Adaptive top_k:    {trace['adaptive_topk_used']}")
        print(f"  Reranking:         {trace['reranking_applied']}")
        print(f"  Compression:       {trace['compression_applied']}")
        print(f"  Confidence:        {trace['retrieval_confidence']:.4f}")
        print(f"  Risk score:        {trace['prediction']['risk_score']}")
        print(f"  Hall. type:        {trace['prediction']['hallucination_type']}")
        print(f"  # Docs:            {len(trace['retrieved_docs'])}")
        print(f"  Answer:            {trace['generation']['answer'][:100]}...")
        if trace["verification"]:
            print(f"  Verdict:           {trace['verification']['verdict']}")
        else:
            print("  Verdict:           (not verified)")
