"""
Tests for modules.strategy_selector.StrategySelector
=====================================================

Covers:
  - Low risk  → direct_llm
  - Medium risk → rag
  - High risk  → rag_verification
  - High-risk type override (mid-score but risky type → rag_verification)
  - Edge cases: boundary values, missing keys, wrong types
"""

from __future__ import annotations

import pytest
from modules.strategy_selector import StrategySelector


@pytest.fixture
def selector() -> StrategySelector:
    return StrategySelector()


# ------------------------------------------------------------------ #
#  Strategy routing                                                    #
# ------------------------------------------------------------------ #

class TestRouting:
    def test_low_risk_direct_llm(self, selector: StrategySelector) -> None:
        result = selector.select({"risk_score": 0.1, "hallucination_type": "none"})
        assert result == "direct_llm"

    def test_medium_risk_rag(self, selector: StrategySelector) -> None:
        # 0.5 is between low (0.3) and high (0.7), type is NOT high-risk
        result = selector.select({"risk_score": 0.5, "hallucination_type": "none"})
        assert result == "rag"

    def test_high_risk_rag_verification(self, selector: StrategySelector) -> None:
        result = selector.select({"risk_score": 0.85, "hallucination_type": "entity"})
        assert result == "rag_verification"

    def test_high_risk_type_overrides(self, selector: StrategySelector) -> None:
        # Mid score (would be 'rag') but citation type bumps to rag_verification
        result = selector.select({"risk_score": 0.4, "hallucination_type": "citation"})
        assert result == "rag_verification"

    def test_all_high_risk_types(self, selector: StrategySelector) -> None:
        for h_type in ["citation", "entity", "relation", "temporal", "reasoning"]:
            result = selector.select({"risk_score": 0.5, "hallucination_type": h_type})
            assert result == "rag_verification", f"Failed for type {h_type}"


# ------------------------------------------------------------------ #
#  Boundary values                                                     #
# ------------------------------------------------------------------ #

class TestBoundaries:
    def test_exactly_low_threshold(self, selector: StrategySelector) -> None:
        # risk_score == 0.3 → NOT < 0.3 → should be rag (type=none)
        result = selector.select({"risk_score": 0.3, "hallucination_type": "none"})
        assert result == "rag"

    def test_exactly_high_threshold(self, selector: StrategySelector) -> None:
        # risk_score == 0.7 → >= 0.7 → rag_verification
        result = selector.select({"risk_score": 0.7, "hallucination_type": "none"})
        assert result == "rag_verification"

    def test_zero_risk(self, selector: StrategySelector) -> None:
        result = selector.select({"risk_score": 0.0, "hallucination_type": "none"})
        assert result == "direct_llm"

    def test_max_risk(self, selector: StrategySelector) -> None:
        result = selector.select({"risk_score": 1.0, "hallucination_type": "none"})
        assert result == "rag_verification"


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_missing_risk_score_raises(self, selector: StrategySelector) -> None:
        with pytest.raises(KeyError, match="risk_score"):
            selector.select({"hallucination_type": "none"})

    def test_non_dict_raises(self, selector: StrategySelector) -> None:
        with pytest.raises(TypeError, match="prediction must be a dict"):
            selector.select("not_a_dict")  # type: ignore[arg-type]

    def test_missing_type_defaults_to_none(self, selector: StrategySelector) -> None:
        # hallucination_type absent → treated as "none"
        result = selector.select({"risk_score": 0.5})
        assert result == "rag"

    def test_result_always_valid_strategy(self, selector: StrategySelector) -> None:
        for score in [0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = selector.select({"risk_score": score, "hallucination_type": "none"})
            assert result in StrategySelector.VALID_STRATEGIES
