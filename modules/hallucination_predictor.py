"""
Hallucination Predictor Module
==============================

Predicts the hallucination risk of a query based on features extracted by
QueryAnalyzer.  Supports two scoring modes:

Feature-only mode (default / fast):
  Uses a scikit-learn logistic regression classifier trained on labelled data.
  Falls back to a heuristic scorer when no trained model exists.

Hybrid self-consistency mode (triggered when query is passed and config enables it):
  Calls Ollama n_samples times at different temperatures and measures how
  consistently the model answers the same question.  High answer variance
  (low embedding similarity) signals uncertainty → high hallucination risk.
  The final risk score is a weighted blend of the feature-based score and the
  inconsistency score derived from self-consistency sampling.

Self-consistency only triggers when:
  1. A query string is passed to predict()
  2. predictor.use_self_consistency: true in config
  3. features["complexity_score"] > predictor.self_consistency_threshold

This keeps latency low for simple queries while applying richer analysis to
complex or high-risk ones.

**Inputs**:  Feature dict from QueryAnalyzer.analyze()
**Outputs**: ``{ risk_score, hallucination_type, type_confidence, hybrid_scoring, ... }``
**Dependencies**: scikit-learn, joblib, numpy, PyYAML, sentence-transformers, ollama
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

# ---------------------------------------------------------------------------
# Lazy-loaded sentence-transformer for self-consistency embedding
# ---------------------------------------------------------------------------

_sc_model = None  # SentenceTransformer instance, shared across calls


def _get_sc_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load a SentenceTransformer for self-consistency scoring.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        SentenceTransformer instance or None if unavailable.
    """
    global _sc_model
    if _sc_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sc_model = SentenceTransformer(model_name)
        except Exception:
            _sc_model = None
    return _sc_model


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

# TruthfulQA category → hallucination type mapping
_CATEGORY_TO_TYPE: Dict[str, str] = {
    "fiction": "entity",
    "misconception": "entity",
    "myth": "entity",
    "conspiracy": "entity",
    "false": "entity",
    "paranormal": "entity",
    "superstition": "entity",
    "quote": "citation",
    "attribution": "citation",
    "history": "temporal",
    "dates": "temporal",
    "timing": "temporal",
    "law": "reasoning",
    "statistics": "reasoning",
    "mathematics": "reasoning",
    "science": "reasoning",
    "logic": "reasoning",
}

# TruthfulQA categories that suggest high hallucination risk (label=1 proxy)
_HIGH_RISK_CATEGORIES = {
    "fiction", "misconception", "myth", "conspiracy", "false belief",
    "paranormal", "superstition", "misattributed", "myths and misconceptions",
}


def _features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert the feature dict from QueryAnalyzer into a flat numpy vector.

    Args:
        features: Dict produced by ``QueryAnalyzer.analyze()``.

    Returns:
        1-D numpy array of shape ``(len(_FEATURE_KEYS),)`` i.e. (11,).
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


def _encode_query(query: str) -> np.ndarray:
    """Encode a query string into a 384-dim sentence embedding.

    Reuses the same all-MiniLM-L6-v2 model as self-consistency scoring to
    avoid loading a second model into memory.  Returns a zero vector if the
    sentence-transformer is unavailable so the pipeline degrades gracefully.

    Args:
        query: Raw query string to encode.

    Returns:
        384-dimensional numpy float64 array (zeros on failure).
    """
    model = _get_sc_model()
    if model is None:
        return np.zeros(384, dtype=np.float64)
    try:
        emb = model.encode([query], normalize_embeddings=True)
        return emb[0].astype(np.float64)
    except Exception:
        return np.zeros(384, dtype=np.float64)


def _features_to_vector_enhanced(
    features: Dict[str, Any],
    query: Optional[str] = None,
) -> np.ndarray:
    """Build a 395-dim feature vector: 11 surface features + 384-dim embedding.

    Concatenates the hand-crafted surface features with the query's sentence
    embedding so the classifier has both linguistic and semantic signal.
    When no query is provided or the sentence transformer is unavailable,
    the embedding portion is padded with zeros to keep the dimension fixed.

    Args:
        features: Feature dict from QueryAnalyzer.analyze().
        query: Optional raw query string for embedding.

    Returns:
        395-dimensional numpy float64 array.
    """
    base = _features_to_vector(features)  # shape (11,)
    emb = _encode_query(query) if query else np.zeros(384, dtype=np.float64)
    return np.concatenate([base, emb])


class HallucinationPredictor:
    """Predicts hallucination risk score and type from query features.

    When no trained model is available on disk the predictor falls back to a
    *heuristic* scoring mode so that the rest of the pipeline remains
    functional before any training data exists.

    When ``query`` is passed to ``predict()`` and self-consistency is enabled
    in config, additional Ollama calls are made to sample n answers at varied
    temperatures.  The embedding variance across answers is blended with the
    feature-based score for a more reliable final risk estimate.
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

    def predict(
        self,
        features: Dict[str, Any],
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict hallucination risk from query features.

        When *query* is provided and self-consistency is enabled in config,
        a hybrid score (feature-based + self-consistency) is computed.
        Otherwise only the feature-based score is returned.

        Args:
            features: Feature dict returned by ``QueryAnalyzer.analyze()``.
            query: Optional raw query string.  Required for self-consistency.

        Returns:
            Dict with:
              ``risk_score`` (float 0-1),
              ``hallucination_type`` (str),
              ``type_confidence`` (float 0-1),
              ``hybrid_scoring`` (bool),
              and optionally ``self_consistency``, ``feature_risk_score``
              when hybrid mode fires.
        """
        if not isinstance(features, dict):
            raise TypeError(f"features must be a dict, got {type(features).__name__}")

        # Base feature-only prediction — query is passed through for embedding
        if self.risk_model is not None and self.type_model is not None:
            base = self._predict_with_model(features, query)
        else:
            base = self._predict_heuristic(features)

        feature_risk = base["risk_score"]

        # Check whether hybrid self-consistency scoring should fire
        pred_cfg = self.config.get("predictor", {})
        use_sc = pred_cfg.get("use_self_consistency", True)
        sc_threshold = float(pred_cfg.get("self_consistency_threshold", 0.3))
        complexity = float(features.get("complexity_score", 0.0))

        if query and use_sc and complexity > sc_threshold:
            n_samples = int(pred_cfg.get("self_consistency_samples", 3))
            sc_result = self.compute_self_consistency(query, n_samples=n_samples)

            w_feat = float(pred_cfg.get("feature_weight", 0.4))
            w_sc = float(pred_cfg.get("self_consistency_weight", 0.6))
            inconsistency = sc_result["inconsistency_score"]

            hybrid_risk = round(w_feat * feature_risk + w_sc * inconsistency, 4)

            return {
                **base,
                "risk_score": hybrid_risk,
                "feature_risk_score": feature_risk,
                "hybrid_scoring": True,
                "self_consistency": sc_result,
            }

        # No hybrid — return base result with hybrid_scoring flag
        return {
            **base,
            "hybrid_scoring": False,
        }

    def compute_self_consistency(
        self,
        query: str,
        n_samples: int = 3,
    ) -> Dict[str, Any]:
        """Sample Ollama n times and measure answer consistency.

        Calls Ollama at temperatures [0.3, 0.6, 0.9] (or linearly spaced
        variants for n_samples != 3) to elicit a range of answers.  Encodes
        all answers with a sentence-transformer and computes the average
        pairwise cosine similarity.

        High similarity → consistent → low hallucination risk.
        Low similarity → inconsistent → high hallucination risk.

        Args:
            query: The user's question to sample.
            n_samples: Number of Ollama calls (default 3).

        Returns:
            Dict with:
              ``consistency_score`` (float 0-1, high = consistent),
              ``inconsistency_score`` (float 0-1, high = inconsistent),
              ``n_samples`` (int, 0 if Ollama unavailable),
              ``answers`` (list[str]).
        """
        _fail = {
            "consistency_score": 1.0,
            "inconsistency_score": 0.0,
            "n_samples": 0,
            "answers": [],
        }

        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "llama3.2")

        # Build temperature list — evenly spaced across [0.3, 0.9]
        if n_samples == 1:
            temperatures = [0.6]
        else:
            step = 0.6 / max(n_samples - 1, 1)
            temperatures = [round(0.3 + i * step, 2) for i in range(n_samples)]

        # Run all Ollama calls in parallel — cuts latency from ~3×t to ~1×t
        def _single_call(args: tuple) -> str:
            _model_name, _query, _temp = args
            try:
                import ollama
                response = ollama.chat(
                    model=_model_name,
                    messages=[{"role": "user", "content": _query}],
                    options={"temperature": _temp},
                )
                return response["message"]["content"].strip()
            except Exception:
                return ""

        args_list = [(model_name, query, t) for t in temperatures]
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(temperatures)) as executor:
                raw_answers = list(executor.map(_single_call, args_list))
        except Exception:
            return _fail

        answers: List[str] = [a for a in raw_answers if a]

        if len(answers) < 2:
            # Not enough answers to compute pairwise similarity
            return {
                "consistency_score": 1.0,
                "inconsistency_score": 0.0,
                "n_samples": len(answers),
                "answers": answers,
            }

        # Encode answers with sentence-transformer
        emb_model = _get_sc_model()
        if emb_model is None:
            return _fail

        try:
            embeddings = emb_model.encode(answers, normalize_embeddings=True)
            # Compute all pairwise cosine similarities (dot product on unit vecs)
            sim_matrix = np.dot(embeddings, embeddings.T)
            # Extract upper-triangle (excluding diagonal)
            n = len(answers)
            sims: List[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(float(sim_matrix[i, j]))

            consistency = float(np.mean(sims)) if sims else 1.0
            consistency = round(min(1.0, max(0.0, consistency)), 4)
            inconsistency = round(1.0 - consistency, 4)

            return {
                "consistency_score": consistency,
                "inconsistency_score": inconsistency,
                "n_samples": len(answers),
                "answers": answers,
            }

        except Exception:
            return _fail

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

        X_base, y_risk, y_type, queries = self._load_dataset(dataset_path)

        if len(X_base) < 10:
            raise ValueError(
                f"Dataset too small ({len(X_base)} samples). Need at least 10."
            )

        # Build enhanced 395-dim feature matrix: 11 surface + 384 query embedding.
        # The sentence-transformer is lazy-loaded; encoding ~800 queries takes ~5–10s.
        print(
            f"  Encoding {len(queries)} queries with all-MiniLM-L6-v2 "
            "(11 surface features + 384-dim embedding) ..."
        )
        enhanced_rows: list[np.ndarray] = []
        for feat_row, query in zip(X_base, queries):
            emb = _encode_query(query) if query else np.zeros(384, dtype=np.float64)
            enhanced_rows.append(np.concatenate([feat_row, emb]))
        X = np.vstack(enhanced_rows)  # shape (n_samples, 395)

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
            pos_idx = (
                list(self.risk_model.classes_).index(1)
                if 1 in self.risk_model.classes_ else 0
            )
            auroc = float(roc_auc_score(yr_test, yr_proba[:, pos_idx]))

        f1 = float(f1_score(yr_test, yr_pred, average="macro", zero_division=0))

        # Save models
        self._save_models()

        return {"auroc": round(auroc, 4), "f1_macro": round(f1, 4)}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _predict_with_model(
        self,
        features: Dict[str, Any],
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict using trained sklearn models.

        Automatically selects the right feature vector dimension based on
        how the loaded model was trained:
          - n_features_in_ == 11  → base 11-dim vector  (old model)
          - n_features_in_ == 395 → enhanced 395-dim vector (new model)

        This ensures backward compatibility with models trained before the
        embedding upgrade.

        Args:
            features: Feature dict from QueryAnalyzer.analyze().
            query: Optional raw query string; used for embedding when the
                   loaded model was trained with enhanced vectors.

        Returns:
            Dict with risk_score, hallucination_type, type_confidence.
        """
        assert self.risk_model is not None and self.type_model is not None

        # Determine expected vector size from the trained model
        expected_dim = getattr(self.risk_model, "n_features_in_", len(_FEATURE_KEYS))
        if expected_dim == len(_FEATURE_KEYS):
            # Legacy 11-dim model — use base features only
            vec = _features_to_vector(features).reshape(1, -1)
        else:
            # Enhanced 395-dim model — concatenate surface features + embedding
            vec = _features_to_vector_enhanced(features, query).reshape(1, -1)

        risk_proba = self.risk_model.predict_proba(vec)[0]
        pos_idx = (
            list(self.risk_model.classes_).index(1)
            if 1 in self.risk_model.classes_ else 0
        )
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
        risk_score = complexity

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
    ) -> tuple[np.ndarray, list[int], list[str], list[str]]:
        """Load labelled JSONL and return X matrix, label vectors, and queries.

        Extracts the "query" field from each record alongside features and
        labels.  The X matrix contains only the 11-dim surface feature vectors;
        enhanced embedding concatenation is handled by train() so that a
        progress message can be shown before the (slow) encoding step.

        Args:
            dataset_path: Path to labelled JSONL file.

        Returns:
            Tuple of (X, y_risk, y_type, queries) where queries is a list of
            raw query strings (empty string when the field is absent).
        """
        X_list: list[np.ndarray] = []
        y_risk: list[int] = []
        y_type: list[str] = []
        queries: list[str] = []

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
                queries.append(record.get("query", ""))

        X = np.vstack(X_list) if X_list else np.empty((0, len(_FEATURE_KEYS)))
        return X, y_risk, y_type, queries

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

    print("=== Hallucination Predictor (feature-only mode) ===")
    result = predictor.predict(sample_features)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== With self-consistency (if Ollama is running) ===")
    result2 = predictor.predict(sample_features, query="What caused the 2008 financial crisis?")
    for k, v in result2.items():
        if k != "self_consistency":
            print(f"  {k}: {v}")
    if result2.get("self_consistency"):
        sc = result2["self_consistency"]
        print(f"  self_consistency.consistency_score: {sc['consistency_score']}")
        print(f"  self_consistency.n_samples: {sc['n_samples']}")
