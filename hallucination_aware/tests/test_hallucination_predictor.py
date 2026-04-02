"""
Tests for modules.hallucination_predictor.HallucinationPredictor
================================================================

Covers:
  - Heuristic prediction happy path (no trained model)
  - Heuristic type assignment for various feature combos
  - Edge case: empty/minimal features
  - Edge case: non-dict input raises TypeError
  - Training on a synthetic dataset and validating metrics structure
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from modules.hallucination_predictor import HallucinationPredictor


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_features(**overrides: Any) -> Dict[str, Any]:
    """Build a default feature dict with optional overrides."""
    base: Dict[str, Any] = {
        "entity_count": 1,
        "query_length_tokens": 10,
        "contains_date": False,
        "contains_citation_pattern": False,
        "multi_hop_indicator": False,
        "entity_type_flags": {
            "PERSON": False, "ORG": False, "LOC": False, "DATE": False,
        },
        "avg_token_length": 4.5,
        "complexity_score": 0.3,
    }
    base.update(overrides)
    return base


def _write_synthetic_dataset(path: str, n: int = 50) -> None:
    """Write a tiny synthetic labelled dataset to *path*."""
    import random
    random.seed(42)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            is_hall = random.random() > 0.5
            h_type = random.choice(["none", "entity", "temporal", "citation", "reasoning", "relation"])
            record = {
                "features": {
                    "entity_count": random.randint(0, 5),
                    "query_length_tokens": random.randint(3, 30),
                    "contains_date": random.random() > 0.5,
                    "contains_citation_pattern": random.random() > 0.7,
                    "multi_hop_indicator": random.random() > 0.6,
                    "entity_type_flags": {
                        "PERSON": random.random() > 0.5,
                        "ORG": random.random() > 0.7,
                        "LOC": random.random() > 0.6,
                        "DATE": random.random() > 0.5,
                    },
                    "avg_token_length": round(random.uniform(3, 8), 2),
                    "complexity_score": round(random.uniform(0, 1), 4),
                },
                "is_hallucination": is_hall,
                "hallucination_type": h_type if is_hall else "none",
            }
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def predictor() -> HallucinationPredictor:
    return HallucinationPredictor()


# ------------------------------------------------------------------ #
#  Happy-path (heuristic mode)                                        #
# ------------------------------------------------------------------ #

class TestHeuristicPredict:
    def test_returns_correct_keys(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(_make_features())
        assert set(result.keys()) == {"risk_score", "hallucination_type", "type_confidence"}

    def test_risk_score_bounded(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(_make_features(complexity_score=0.95))
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_citation_type(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(_make_features(contains_citation_pattern=True))
        assert result["hallucination_type"] == "citation"

    def test_temporal_type(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(_make_features(contains_date=True))
        assert result["hallucination_type"] == "temporal"

    def test_reasoning_type(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(
            _make_features(multi_hop_indicator=True, contains_date=False)
        )
        assert result["hallucination_type"] == "reasoning"

    def test_entity_type(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict(
            _make_features(entity_count=3, multi_hop_indicator=False)
        )
        assert result["hallucination_type"] == "entity"


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_features(self, predictor: HallucinationPredictor) -> None:
        result = predictor.predict({})
        assert "risk_score" in result

    def test_non_dict_raises(self, predictor: HallucinationPredictor) -> None:
        with pytest.raises(TypeError, match="features must be a dict"):
            predictor.predict("not a dict")  # type: ignore[arg-type]

    def test_missing_entity_type_flags(self, predictor: HallucinationPredictor) -> None:
        feats = _make_features()
        del feats["entity_type_flags"]
        result = predictor.predict(feats)
        assert 0.0 <= result["risk_score"] <= 1.0


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

class TestTraining:
    def test_train_returns_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = str(Path(tmpdir) / "dataset.jsonl")
            _write_synthetic_dataset(ds_path, n=60)

            pred = HallucinationPredictor()
            # Override model save paths to temp dir
            pred.model_path = str(Path(tmpdir) / "risk.pkl")
            pred.type_model_path = str(Path(tmpdir) / "type.pkl")

            metrics = pred.train(ds_path)
            assert "auroc" in metrics
            assert "f1_macro" in metrics
            assert 0.0 <= metrics["auroc"] <= 1.0
            assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_train_file_not_found(self, predictor: HallucinationPredictor) -> None:
        with pytest.raises(FileNotFoundError):
            predictor.train("nonexistent.jsonl")

    def test_train_too_few_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = str(Path(tmpdir) / "tiny.jsonl")
            _write_synthetic_dataset(ds_path, n=5)
            pred = HallucinationPredictor()
            pred.model_path = str(Path(tmpdir) / "r.pkl")
            pred.type_model_path = str(Path(tmpdir) / "t.pkl")
            with pytest.raises(ValueError, match="too small"):
                pred.train(ds_path)
