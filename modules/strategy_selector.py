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

    def select(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any] | None = None,
    ) -> str:
        """Select the answering strategy for a given prediction.

        Decision logic (in priority order):

        HARD OVERRIDES (applied first when *features* is provided):
          a. Citation pattern present              → ``'rag_verification'``
          b. Multi-hop AND complexity > 0.6        → ``'rag_verification'``
          c. Complexity > 0.75                     → ``'rag_verification'``

        RISK-SCORE THRESHOLDS (applied after hard overrides):
          1. ``risk_score < low_threshold``        → ``'direct_llm'``
          2. ``risk_score >= high_threshold`` OR
             ``hallucination_type`` in high_risk_types → ``'rag_verification'``
          3. Otherwise                             → ``'rag'``

        Hard overrides exist because the ML model tends to underestimate risk
        for factually complex queries (multi-hop, citation-heavy, high-complexity).
        They ensure these queries always get retrieved context and verification
        regardless of the predicted risk score.

        Args:
            prediction: Dict with at least ``risk_score`` (float) and
                        ``hallucination_type`` (str).
            features: Optional feature dict from QueryAnalyzer.  Required to
                      activate the hard-override rules.

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

        # ── Hard overrides (feature-driven, applied before risk score) ────
        if features is not None:
            complexity: float = float(features.get("complexity_score", 0.0))
            multi_hop: bool = bool(features.get("multi_hop_indicator", False))
            citation: bool = bool(features.get("contains_citation_pattern", False))

            # Citation queries always need grounding + verification
            if citation:
                return "rag_verification"

            # Multi-hop + high complexity → grounding + verification
            if multi_hop and complexity > 0.6:
                return "rag_verification"

            # Very high complexity alone → grounding + verification
            if complexity > 0.75:
                return "rag_verification"

        # ── Risk-score threshold logic ────────────────────────────────────
        risk_score: float = float(prediction["risk_score"])
        h_type: str = prediction.get("hallucination_type", "none")

        # Low risk → direct LLM
        if risk_score < self.low_threshold:
            return "direct_llm"

        # High risk or dangerous type → RAG + verification
        if risk_score >= self.high_threshold or h_type in self.high_risk_types:
            return "rag_verification"

        # Medium risk → RAG only
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
