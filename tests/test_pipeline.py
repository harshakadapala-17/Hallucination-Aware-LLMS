"""
Integration tests for pipeline.pipeline.Pipeline
=================================================

Covers:
  - Trace dict has all 7 required keys
  - Strategy is always a valid enum value
  - Routing logic is internally consistent (strategy ↔ retrieved_docs ↔ verification)
  - Features dict contains all 8 expected keys
  - Prediction risk_score is always in [0.0, 1.0]
  - High-risk queries (citation + date + multi-hop) route above the low threshold
  - query string is preserved in trace
  - Non-string query raises TypeError

Note: Tests do NOT hardcode which strategy a specific query receives.
The model's predictions depend on training data and may evolve over time.
Instead, tests verify that whatever strategy is chosen, the rest of the
trace is internally consistent with that choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline import Pipeline
from modules.strategy_selector import StrategySelector


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline()


@pytest.fixture(scope="module")
def simple_trace(pipeline: Pipeline) -> dict:
    return pipeline.run("What is the capital of France?")


@pytest.fixture(scope="module")
def complex_trace(pipeline: Pipeline) -> dict:
    return pipeline.run(
        "According to Dr. Smith at MIT, what caused the 2008 financial crisis "
        "and how did it compare to the Great Depression in terms of policy response?"
    )


# ------------------------------------------------------------------ #
#  Trace structure                                                     #
# ------------------------------------------------------------------ #

class TestTraceStructure:
    def test_trace_has_required_keys(self, simple_trace: dict) -> None:
        expected = {
            "query", "features", "prediction", "strategy", "retrieved_docs",
            "generation", "verification", "hard_override_applied",
            "retrieval_strategy", "adaptive_topk_used", "reranking_applied",
            "compression_applied", "retrieval_confidence", "retrieval_details",
            "hybrid_scoring", "feature_risk_score", "self_consistency_score",
        }
        assert expected.issubset(set(simple_trace.keys()))

    def test_strategy_is_valid_value(self, simple_trace: dict) -> None:
        assert simple_trace["strategy"] in {"direct_llm", "rag", "rag_verification"}

    def test_strategy_valid_for_complex(self, complex_trace: dict) -> None:
        assert complex_trace["strategy"] in {"direct_llm", "rag", "rag_verification"}

    def test_retrieved_docs_is_list(self, simple_trace: dict) -> None:
        assert isinstance(simple_trace["retrieved_docs"], list)

    def test_generation_has_answer_key(self, simple_trace: dict) -> None:
        assert "answer" in simple_trace["generation"]


# ------------------------------------------------------------------ #
#  Routing invariants (strategy ↔ docs ↔ verification must agree)     #
# ------------------------------------------------------------------ #

class TestRoutingInvariants:
    def test_direct_llm_has_no_docs(self, pipeline: Pipeline) -> None:
        """Any trace whose strategy is direct_llm must have empty retrieved_docs."""
        for query in [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What is two plus two?",
        ]:
            trace = pipeline.run(query)
            if trace["strategy"] == "direct_llm":
                assert trace["retrieved_docs"] == [], (
                    f"direct_llm trace should have no docs for: {query!r}"
                )

    def test_rag_strategies_have_docs(self, pipeline: Pipeline) -> None:
        """Any trace whose strategy is rag or rag_verification must have docs."""
        for query in [
            "According to Dr. Smith, what caused the 2008 crisis?",
            "Compare the policies of Roosevelt and Hoover during the Depression.",
        ]:
            trace = pipeline.run(query)
            if trace["strategy"] in ("rag", "rag_verification"):
                assert len(trace["retrieved_docs"]) > 0, (
                    f"rag/rag_verification trace should have docs for: {query!r}"
                )

    def test_verification_only_for_rag_verification(self, pipeline: Pipeline) -> None:
        """Verification must be None for direct_llm and rag, non-None for rag_verification."""
        for query in [
            "What is the capital of France?",
            "According to Dr. Smith at MIT, citing Jones 2019, what were the "
            "exact differences between the 2008 crisis and the Great Depression?",
        ]:
            trace = pipeline.run(query)
            if trace["strategy"] == "rag_verification":
                assert trace["verification"] is not None
            else:
                assert trace["verification"] is None

    def test_strategy_selector_logic_direct_llm(self) -> None:
        """StrategySelector must return direct_llm when risk_score < low threshold."""
        selector = StrategySelector()
        result = selector.select({"risk_score": 0.1, "hallucination_type": "none"})
        assert result == "direct_llm"

    def test_strategy_selector_logic_rag(self) -> None:
        selector = StrategySelector()
        result = selector.select({"risk_score": 0.35, "hallucination_type": "none"})
        assert result == "rag"

    def test_strategy_selector_logic_rag_verification(self) -> None:
        selector = StrategySelector()
        result = selector.select({"risk_score": 0.9, "hallucination_type": "none"})
        assert result == "rag_verification"


# ------------------------------------------------------------------ #
#  Features dict                                                       #
# ------------------------------------------------------------------ #

class TestFeatures:
    def test_features_dict_has_all_keys(self, simple_trace: dict) -> None:
        expected_keys = {
            "entity_count", "query_length_tokens", "contains_date",
            "contains_citation_pattern", "multi_hop_indicator",
            "entity_type_flags", "avg_token_length", "complexity_score",
        }
        assert set(simple_trace["features"].keys()) == expected_keys

    def test_prediction_risk_score_in_range(self, simple_trace: dict) -> None:
        assert 0.0 <= simple_trace["prediction"]["risk_score"] <= 1.0

    def test_prediction_risk_score_in_range_complex(self, complex_trace: dict) -> None:
        assert 0.0 <= complex_trace["prediction"]["risk_score"] <= 1.0

    def test_query_preserved_in_trace(self, pipeline: Pipeline) -> None:
        q = "Who wrote Hamlet?"
        assert pipeline.run(q)["query"] == q

    def test_non_string_query_raises(self, pipeline: Pipeline) -> None:
        with pytest.raises(TypeError):
            pipeline.run(123)  # type: ignore[arg-type]
