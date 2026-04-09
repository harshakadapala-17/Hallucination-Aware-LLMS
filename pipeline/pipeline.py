"""
Pipeline Module
===============

Orchestrates the full Hallucination-Aware Adaptive LLM pipeline:
  QueryAnalyzer → HallucinationPredictor → StrategySelector →
  (RAGModule) → GenerationModule → (VerificationModule)

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
          4. Retrieve documents (if strategy requires RAG).
          5. Generate answer.
          6. Verify answer (if strategy is rag_verification).
          7. Return the full trace dict.

        Args:
            query: The user's question.

        Returns:
            Trace dict with keys: ``query``, ``features``, ``prediction``,
            ``strategy``, ``retrieved_docs``, ``generation``, ``verification``.

        Raises:
            TypeError: If *query* is not a str.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        # Step 1: Analyse
        features: Dict[str, Any] = self.query_analyzer.analyze(query)

        # Step 2: Predict
        prediction: Dict[str, Any] = self.predictor.predict(features)

        # Step 3: Select strategy
        strategy: str = self.selector.select(prediction)

        # Step 4: Retrieve (only for rag / rag_verification)
        retrieved_docs: List[Dict[str, Any]] = []
        if strategy in ("rag", "rag_verification"):
            top_k = self.config.get("rag", {}).get("top_k", 3)
            retrieved_docs = self.rag.retrieve(query, top_k=top_k)

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
        print('='*70)
        trace = pipeline.run(q)
        print(f"  Strategy:    {trace['strategy']}")
        print(f"  Risk score:  {trace['prediction']['risk_score']}")
        print(f"  Hall. type:  {trace['prediction']['hallucination_type']}")
        print(f"  # Docs:      {len(trace['retrieved_docs'])}")
        print(f"  Answer:      {trace['generation']['answer'][:100]}...")
        if trace["verification"]:
            print(f"  Verdict:     {trace['verification']['verdict']}")
        else:
            print("  Verdict:     (not verified)")
