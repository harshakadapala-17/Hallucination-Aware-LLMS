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
          1. Analyse query features.
          2. Predict hallucination risk.
          3. Select answering strategy.
          4. Unified agent loop (max_agent_iterations):
               a. Retrieve documents (adaptive top_k, routing by complexity).
               b. Re-rank with cross-encoder.
               c. Check retrieval confidence; reformulate + retry if low.
               d. Compress context.
               e. Generate answer.
               f. Verify answer (if strategy is rag_verification).
               g. If refuted: reformulate query + retry full loop.
               h. If supported / partially_supported / last iteration: stop.
          5. Return full trace dict with agent loop metadata.

        Args:
            query: The user's question.

        Returns:
            Trace dict with all pipeline fields plus agentic fields:
            agent_iterations, agent_log, final_query_used,
            query_reformulated.

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

        # ── Agentic loop state ────────────────────────────────────────────
        rag_cfg = self.config.get("rag", {})
        max_iterations: int = int(rag_cfg.get("max_agent_iterations", 2))
        iteration: int = 0
        current_query: str = query   # may be reformulated across iterations
        agent_log: List[Dict[str, Any]] = []
        generation: Dict[str, Any] = {"answer": "", "strategy_used": strategy}
        verification: Optional[Dict[str, Any]] = None

        # ── Short-circuit: direct_llm needs no retrieval or agent loop ────
        if strategy == "direct_llm":
            generation = self.generator.generate(query, "direct_llm", context=[])
        else:
            # ── Unified agent loop ────────────────────────────────────────
            # Each iteration attempts retrieval → generation → verification.
            # If the attempt fails (low confidence or refuted) the query is
            # reformulated via Ollama and the loop restarts from retrieval.
            # Hard max of max_agent_iterations prevents infinite loops.
            complexity = float(features.get("complexity_score", 0.5))
            multi_hop = bool(features.get("multi_hop_indicator", False))
            citation = bool(features.get("contains_citation_pattern", False))
            high_complexity = complexity > 0.7
            use_decomp = rag_cfg.get("use_query_decomposition", True)
            use_reranking = rag_cfg.get("use_reranking", True)
            use_compression = rag_cfg.get("use_contextual_compression", True)
            min_conf = float(rag_cfg.get("min_retrieval_confidence", 0.3))

            # Adaptive top_k is fixed per query (complexity doesn't change)
            if rag_cfg.get("use_adaptive_topk", True):
                top_k = self.rag.get_adaptive_topk(complexity)
            else:
                top_k = int(rag_cfg.get("top_k", 3))
            adaptive_topk_used = top_k

            while iteration < max_iterations:
                iteration += 1

                # ── 4a: Choose retrieval method for current_query ─────────
                if use_decomp and (citation or high_complexity):
                    retrieved_docs = self.rag.retrieve_decomposed(
                        current_query, top_k=top_k
                    )
                    retrieval_strategy = "decomposed"
                    retrieval_details["sub_questions"] = list(
                        getattr(self.rag, "_last_sub_questions", [])
                    )
                elif multi_hop:
                    retrieved_docs = self.rag.retrieve_multihop(
                        current_query, top_k=top_k
                    )
                    retrieval_strategy = "multihop"
                    retrieval_details["follow_up_query"] = getattr(
                        self.rag, "_last_followup_query", ""
                    )
                else:
                    retrieved_docs = self.rag.retrieve(current_query, top_k=top_k)
                    retrieval_strategy = "standard"

                # ── 4b: Cross-encoder re-ranking ──────────────────────────
                if use_reranking and retrieved_docs:
                    retrieved_docs = self.rag.rerank(current_query, retrieved_docs)
                    reranking_applied = True

                # ── 4c: Retrieval confidence check ────────────────────────
                retrieval_confidence = self.rag.get_retrieval_confidence(retrieved_docs)

                if retrieved_docs and retrieval_confidence < min_conf and not hard_override:
                    if iteration < max_iterations:
                        # Confidence too low — reformulate query and retry
                        reformulated = self.rag.reformulate_query(
                            query,
                            context="No relevant documents found",
                            reason="low_retrieval_confidence",
                        )
                        agent_log.append({
                            "iteration": iteration,
                            "action": "reformulate_query",
                            "reason": "low_retrieval_confidence",
                            "original_query": current_query,
                            "reformulated_query": reformulated,
                        })
                        current_query = reformulated
                        # Reset retrieved_docs for the next iteration
                        retrieved_docs = []
                        continue  # restart loop with reformulated query
                    else:
                        # Last iteration — give up on RAG, fall back to direct_llm
                        strategy = "direct_llm"
                        retrieved_docs = []
                        retrieval_details["confidence_fallback"] = True
                elif retrieved_docs and retrieval_confidence < min_conf and hard_override:
                    # Hard override active — keep docs despite low confidence
                    retrieval_details["low_confidence_override"] = True

                # ── 4d: Contextual compression ────────────────────────────
                if use_compression and retrieved_docs:
                    retrieved_docs = self.rag.compress_context(
                        current_query, retrieved_docs
                    )
                    compression_applied = True

                # ── 5: Generate ───────────────────────────────────────────
                context_texts = [doc["text"] for doc in retrieved_docs]
                generation = self.generator.generate(
                    current_query, strategy, context=context_texts
                )

                # ── 6: Verify (only for rag_verification) ─────────────────
                verification = None
                if strategy == "rag_verification" and retrieved_docs:
                    verification = self.verifier.verify(
                        generation["answer"], retrieved_docs
                    )
                    verdict = verification.get("verdict")

                    if verdict == "refuted" and iteration < max_iterations:
                        # Refuted — reformulate query and retry the full loop
                        reformulated = self.rag.reformulate_query(
                            query,
                            context=generation["answer"][:200],
                            reason="refuted",
                        )
                        agent_log.append({
                            "iteration": iteration,
                            "action": "reformulate_and_retry",
                            "reason": "refuted",
                            "original_query": current_query,
                            "reformulated_query": reformulated,
                        })
                        current_query = reformulated
                        continue  # restart loop with reformulated query

                # Supported, partially_supported, unverifiable, or last
                # iteration — accept this result and exit the loop.
                break

        # ── Step 5: Assemble trace ────────────────────────────────────────
        query_reformulated: bool = current_query != query

        trace: Dict[str, Any] = {
            "query": query,                          # always the original
            "features": features,
            "prediction": prediction,
            "strategy": strategy,
            "retrieved_docs": retrieved_docs,
            "generation": generation,
            "verification": verification,
            # ── Retrieval intelligence ─────────────────────────────────
            "retrieval_strategy": retrieval_strategy,
            "adaptive_topk_used": adaptive_topk_used,
            "reranking_applied": reranking_applied,
            "compression_applied": compression_applied,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_details": retrieval_details,
            # ── Hard override ──────────────────────────────────────────
            "hard_override_applied": hard_override,
            # ── Hybrid scoring ─────────────────────────────────────────
            "hybrid_scoring": prediction.get("hybrid_scoring", False),
            "self_consistency_score": (
                prediction.get("self_consistency", {}).get("consistency_score")
                if prediction.get("hybrid_scoring") else None
            ),
            "feature_risk_score": prediction.get("feature_risk_score"),
            # ── Agent loop metadata ────────────────────────────────────
            "agent_iterations": iteration if strategy != "direct_llm" else 0,
            "agent_log": agent_log,
            "final_query_used": current_query,
            "query_reformulated": query_reformulated,
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
        print(f"  Agent iterations:  {trace['agent_iterations']}")
        print(f"  Query reformulated:{trace['query_reformulated']}")
        if trace["query_reformulated"]:
            print(f"  Final query:       {trace['final_query_used'][:80]}")
