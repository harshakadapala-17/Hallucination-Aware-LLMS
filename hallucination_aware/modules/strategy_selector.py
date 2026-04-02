"""
Strategy Selector Module
========================

Routes a query to the safest answering strategy based on the predicted
hallucination risk score and type.

**Inputs**:  Prediction dict from ``HallucinationPredictor.predict()``.
**Outputs**: One of ``'direct_llm'`` | ``'rag'`` | ``'rag_verification'``.
**Dependencies**: PyYAML (via load_config) — no other project modules.
"""

from __future__ import annotations

from typing import Any, Dict

from modules import load_config


class StrategySelector:
    """Select an answering strategy based on hallucination risk."""

    VALID_STRATEGIES = {"direct_llm", "rag", "rag_verification"}

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the StrategySelector.

        Args:
            config: Pre-loaded configuration dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()
        strat_cfg = self.config.get("strategy", {})
        thresholds = strat_cfg.get("thresholds", {})

        self.low_threshold: float = float(thresholds.get("low", 0.3))
        self.high_threshold: float = float(thresholds.get("high", 0.7))
        self.high_risk_types: list[str] = strat_cfg.get(
            "high_risk_types",
            ["citation", "entity", "relation", "temporal", "reasoning"],
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def select(self, prediction: Dict[str, Any]) -> str:
        """Select the answering strategy for a given prediction.

        Decision logic:
          1. ``risk_score < low_threshold``  →  ``'direct_llm'``
          2. ``risk_score >= high_threshold`` **or** the predicted
             ``hallucination_type`` is in ``high_risk_types``
             →  ``'rag_verification'``
          3. Otherwise                       →  ``'rag'``

        Args:
            prediction: Dict with at least ``risk_score`` (float) and
                        ``hallucination_type`` (str).

        Returns:
            One of ``'direct_llm'``, ``'rag'``, or ``'rag_verification'``.

        Raises:
            TypeError: If *prediction* is not a dict.
            KeyError: If ``risk_score`` is missing.
        """
        if not isinstance(prediction, dict):
            raise TypeError(
                f"prediction must be a dict, got {type(prediction).__name__}"
            )
        if "risk_score" not in prediction:
            raise KeyError("prediction dict must contain 'risk_score'")

        risk_score: float = float(prediction["risk_score"])
        h_type: str = prediction.get("hallucination_type", "none")

        # Low risk  → direct LLM
        if risk_score < self.low_threshold:
            return "direct_llm"

        # High risk or dangerous type  → RAG + verification
        if risk_score >= self.high_threshold or h_type in self.high_risk_types:
            return "rag_verification"

        # Medium risk  → RAG only
        return "rag"


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    selector = StrategySelector()

    test_predictions = [
        {"risk_score": 0.1, "hallucination_type": "none", "type_confidence": 0.9},
        {"risk_score": 0.5, "hallucination_type": "none", "type_confidence": 0.6},
        {"risk_score": 0.85, "hallucination_type": "entity", "type_confidence": 0.8},
        {"risk_score": 0.4, "hallucination_type": "citation", "type_confidence": 0.7},
    ]

    print("=== Strategy Selector ===")
    for pred in test_predictions:
        strategy = selector.select(pred)
        print(f"  risk={pred['risk_score']:.2f}  type={pred['hallucination_type']:<10}  → {strategy}")
