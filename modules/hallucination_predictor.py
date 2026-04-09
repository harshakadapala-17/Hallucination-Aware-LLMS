"""
Hallucination Predictor Module
==============================

Predicts the hallucination risk of a query based on features extracted by
QueryAnalyzer. Uses a scikit-learn logistic regression (or XGBoost) classifier
trained on labelled data.

**Inputs**:  Feature dict from QueryAnalyzer.analyze()
**Outputs**: ``{ risk_score: float, hallucination_type: str, type_confidence: float }``
**Dependencies**: scikit-learn, joblib, numpy, PyYAML — no other project modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from modules import load_config


# ----------------------------------------------------------------------- #
#  Feature vector helpers                                                  #
# ----------------------------------------------------------------------- #

# Canonical order for numeric feature vector
_FEATURE_KEYS: List[str] = [
    "entity_count",
    "query_length_tokens",
    "contains_date",
    "contains_citation_pattern",
    "multi_hop_indicator",
    "avg_token_length",
    "complexity_score",
    "entity_type_PERSON",
    "entity_type_ORG",
    "entity_type_LOC",
    "entity_type_DATE",
]

# Canonical hallucination type labels
_HALLUCINATION_TYPES: List[str] = [
    "none",
    "citation",
    "entity",
    "relation",
    "temporal",
    "reasoning",
]


def _features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert the feature dict from QueryAnalyzer into a flat numpy vector.

    Args:
        features: Dict produced by ``QueryAnalyzer.analyze()``.

    Returns:
        1-D numpy array of shape ``(len(_FEATURE_KEYS),)``.
    """
    entity_flags = features.get("entity_type_flags", {})
    vec = [
        float(features.get("entity_count", 0)),
        float(features.get("query_length_tokens", 0)),
        float(features.get("contains_date", False)),
        float(features.get("contains_citation_pattern", False)),
        float(features.get("multi_hop_indicator", False)),
        float(features.get("avg_token_length", 0.0)),
        float(features.get("complexity_score", 0.0)),
        float(entity_flags.get("PERSON", False)),
        float(entity_flags.get("ORG", False)),
        float(entity_flags.get("LOC", False)),
        float(entity_flags.get("DATE", False)),
    ]
    return np.array(vec, dtype=np.float64)


class HallucinationPredictor:
    """Predicts hallucination risk score and type from query features.

    When no trained model is available on disk the predictor falls back to a
    *heuristic* scoring mode so that the rest of the pipeline remains
    functional before any training data exists.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the predictor, optionally loading saved models.

        Args:
            config: Pre-loaded configuration dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()
        pred_cfg = self.config.get("predictor", {})

        self.model_path: str = pred_cfg.get("model_path", "data/predictor.pkl")
        self.type_model_path: str = pred_cfg.get("type_model_path", "data/type_classifier.pkl")
        self.test_size: float = pred_cfg.get("test_size", 0.2)
        self.random_seed: int = pred_cfg.get("random_seed", 42)

        self.risk_model: Optional[LogisticRegression] = None
        self.type_model: Optional[LogisticRegression] = None

        # Try to load pre-trained models
        self._try_load_models()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict hallucination risk from query features.

        Args:
            features: Feature dict returned by ``QueryAnalyzer.analyze()``.

        Returns:
            Dict with ``risk_score`` (float 0-1), ``hallucination_type``
            (str), and ``type_confidence`` (float 0-1).
        """
        if not isinstance(features, dict):
            raise TypeError(f"features must be a dict, got {type(features).__name__}")

        vec = _features_to_vector(features).reshape(1, -1)

        if self.risk_model is not None and self.type_model is not None:
            return self._predict_with_model(vec)
        return self._predict_heuristic(features)

    def train(self, dataset_path: str) -> Dict[str, float]:
        """Train risk and type classifiers from a labelled JSONL dataset.

        Each line in the file must be a JSON object with:
          - ``features``: dict (QueryAnalyzer output)
          - ``is_hallucination``: bool
          - ``hallucination_type``: str (one of _HALLUCINATION_TYPES)

        Args:
            dataset_path: Path to the labelled JSONL file.

        Returns:
            Dict with ``auroc`` and ``f1_macro`` evaluation metrics.

        Raises:
            FileNotFoundError: If dataset_path does not exist.
            ValueError: If dataset has fewer than 10 samples.
        """
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        X, y_risk, y_type = self._load_dataset(dataset_path)

        if len(X) < 10:
            raise ValueError(
                f"Dataset too small ({len(X)} samples). Need at least 10."
            )

        # Split
        X_train, X_test, yr_train, yr_test, yt_train, yt_test = train_test_split(
            X, y_risk, y_type,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y_risk if len(set(y_risk)) > 1 else None,
        )

        # Train risk model (binary)
        self.risk_model = LogisticRegression(
            random_state=self.random_seed, max_iter=1000
        )
        self.risk_model.fit(X_train, yr_train)

        # Train type model (multiclass)
        self.type_model = LogisticRegression(
            random_state=self.random_seed, max_iter=1000
        )
        self.type_model.fit(X_train, yt_train)

        # Evaluate
        yr_pred = self.risk_model.predict(X_test)
        yr_proba = self.risk_model.predict_proba(X_test)

        # AUROC — handle single-class edge case
        unique_classes = sorted(set(yr_test))
        if len(unique_classes) < 2:
            auroc = 0.0
        else:
            pos_idx = list(self.risk_model.classes_).index(1) if 1 in self.risk_model.classes_ else 0
            auroc = float(roc_auc_score(yr_test, yr_proba[:, pos_idx]))

        f1 = float(f1_score(yr_test, yr_pred, average="macro", zero_division=0))

        # Save models
        self._save_models()

        return {"auroc": round(auroc, 4), "f1_macro": round(f1, 4)}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _predict_with_model(self, vec: np.ndarray) -> Dict[str, Any]:
        """Predict using trained sklearn models."""
        assert self.risk_model is not None and self.type_model is not None

        risk_proba = self.risk_model.predict_proba(vec)[0]
        pos_idx = list(self.risk_model.classes_).index(1) if 1 in self.risk_model.classes_ else 0
        risk_score = float(risk_proba[pos_idx])

        type_proba = self.type_model.predict_proba(vec)[0]
        type_idx = int(np.argmax(type_proba))
        hallucination_type = self.type_model.classes_[type_idx]
        type_confidence = float(type_proba[type_idx])

        return {
            "risk_score": round(risk_score, 4),
            "hallucination_type": str(hallucination_type),
            "type_confidence": round(type_confidence, 4),
        }

    @staticmethod
    def _predict_heuristic(features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic when no trained model is available.

        Uses the complexity_score directly as the risk proxy.
        """
        complexity = float(features.get("complexity_score", 0.0))
        risk_score = complexity  # direct proxy

        # Heuristic type assignment (priority order)
        entity_count = features.get("entity_count", 0)
        contains_citation = features.get("contains_citation_pattern", False)
        contains_date = features.get("contains_date", False)
        multi_hop = features.get("multi_hop_indicator", False)
        
        if contains_citation:
            h_type = "citation"
        elif contains_date:
            h_type = "temporal"
        elif multi_hop:
            h_type = "reasoning"
        elif entity_count >= 2:
            h_type = "entity"
        else:
            h_type = "none"

        return {
            "risk_score": round(risk_score, 4),
            "hallucination_type": h_type,
            "type_confidence": round(complexity * 0.8, 4),
        }

    def _load_dataset(
        self, dataset_path: str
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Load labelled JSONL and return X matrix + label vectors."""
        X_list: list[np.ndarray] = []
        y_risk: list[int] = []
        y_type: list[str] = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                feats = record["features"]
                X_list.append(_features_to_vector(feats))
                y_risk.append(int(record.get("is_hallucination", False)))
                y_type.append(record.get("hallucination_type", "none"))

        X = np.vstack(X_list) if X_list else np.empty((0, len(_FEATURE_KEYS)))
        return X, y_risk, y_type

    def _try_load_models(self) -> None:
        """Attempt to load saved models from disk (silent on failure)."""
        if Path(self.model_path).exists():
            try:
                self.risk_model = joblib.load(self.model_path)
            except Exception:
                self.risk_model = None
        if Path(self.type_model_path).exists():
            try:
                self.type_model = joblib.load(self.type_model_path)
            except Exception:
                self.type_model = None

    def _save_models(self) -> None:
        """Persist trained models to disk."""
        if self.risk_model is not None:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.risk_model, self.model_path)
        if self.type_model is not None:
            Path(self.type_model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.type_model, self.type_model_path)


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    predictor = HallucinationPredictor()

    sample_features = {
        "entity_count": 2,
        "query_length_tokens": 15,
        "contains_date": True,
        "contains_citation_pattern": False,
        "multi_hop_indicator": True,
        "entity_type_flags": {"PERSON": True, "ORG": False, "LOC": False, "DATE": True},
        "avg_token_length": 5.2,
        "complexity_score": 0.72,
    }

    print("=== Hallucination Predictor (heuristic mode) ===")
    result = predictor.predict(sample_features)
    for k, v in result.items():
        print(f"  {k}: {v}")

    minimal_features = {
        "entity_count": 0,
        "query_length_tokens": 3,
        "contains_date": False,
        "contains_citation_pattern": False,
        "multi_hop_indicator": False,
        "entity_type_flags": {"PERSON": False, "ORG": False, "LOC": False, "DATE": False},
        "avg_token_length": 4.0,
        "complexity_score": 0.1,
    }
    print("\n=== Low-risk query ===")
    result2 = predictor.predict(minimal_features)
    for k, v in result2.items():
        print(f"  {k}: {v}")
