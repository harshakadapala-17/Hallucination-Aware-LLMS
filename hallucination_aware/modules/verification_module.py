"""
Verification Module
===================

Post-generation fact-checking that compares an LLM answer against retrieved
evidence to flag unsupported or refuted claims.

**Inputs**:  Answer string + list of evidence dicts from RAGModule.
**Outputs**: ``{ verdict: str, verified_claims: List, flagged_claims: List }``
**Dependencies**: sentence-transformers (via shared encoder), numpy, PyYAML.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from modules import load_config

# Lazy imports
_SentenceTransformer = None


def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class VerificationModule:
    """Verify an LLM-generated answer against retrieved evidence."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the VerificationModule.

        Args:
            config: Pre-loaded configuration dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()
        ver_cfg = self.config.get("verification", {})

        self.sim_threshold_supported: float = float(
            ver_cfg.get("sim_threshold_supported", 0.65)
        )
        self.sim_threshold_refuted: float = float(
            ver_cfg.get("sim_threshold_refuted", 0.40)
        )
        self.max_claims: int = int(ver_cfg.get("max_claims", 5))
        self.use_llm_entailment: bool = bool(
            ver_cfg.get("use_llm_entailment", False)
        )

        rag_cfg = self.config.get("rag", {})
        self._embedding_model_name: str = rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def verify(
        self, answer: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify an answer against evidence documents.

        Steps:
          1. Split *answer* into individual claims (sentence-level).
          2. For each claim, compute cosine similarity against all evidence.
          3. Classify claim as *supported*, *refuted*, or *unverifiable*.
          4. Aggregate into a final verdict.

        Args:
            answer: The LLM-generated answer text.
            evidence: List of ``{text, source, score}`` dicts from RAGModule.

        Returns:
            Dict with ``verdict`` (str: 'supported' | 'partially_supported'
            | 'refuted' | 'unverifiable'), ``verified_claims`` (list),
            and ``flagged_claims`` (list).

        Raises:
            TypeError: If *answer* is not a str or *evidence* is not a list.
        """
        if not isinstance(answer, str):
            raise TypeError(f"answer must be a str, got {type(answer).__name__}")
        if not isinstance(evidence, list):
            raise TypeError(f"evidence must be a list, got {type(evidence).__name__}")

        claims = self._extract_claims(answer)

        if not claims:
            return {
                "verdict": "unverifiable",
                "verified_claims": [],
                "flagged_claims": [],
            }

        if not evidence:
            return {
                "verdict": "unverifiable",
                "verified_claims": [],
                "flagged_claims": [
                    {"claim": c, "status": "unverifiable", "max_similarity": 0.0}
                    for c in claims
                ],
            }

        evidence_texts = [e.get("text", "") for e in evidence]
        verified_claims: List[Dict[str, Any]] = []
        flagged_claims: List[Dict[str, Any]] = []

        for claim in claims:
            max_sim = self._max_similarity(claim, evidence_texts)
            status = self._classify_claim(max_sim)

            entry = {
                "claim": claim,
                "status": status,
                "max_similarity": round(max_sim, 4),
            }

            if status == "supported":
                verified_claims.append(entry)
            else:
                flagged_claims.append(entry)

        verdict = self._aggregate_verdict(verified_claims, flagged_claims)

        return {
            "verdict": verdict,
            "verified_claims": verified_claims,
            "flagged_claims": flagged_claims,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _extract_claims(self, answer: str) -> List[str]:
        """Split an answer into individual claims (sentences).

        Limits to ``self.max_claims`` sentences.

        Args:
            answer: Raw answer text.

        Returns:
            List of claim strings.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return claims[: self.max_claims]

    def _max_similarity(self, claim: str, evidence_texts: List[str]) -> float:
        """Return the maximum cosine similarity between *claim* and evidence.

        Args:
            claim: Single claim string.
            evidence_texts: List of evidence text strings.

        Returns:
            Maximum cosine similarity (float).
        """
        model = self._get_model()
        all_texts = [claim] + evidence_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        claim_vec = embeddings[0:1]
        evidence_vecs = embeddings[1:]
        similarities = np.dot(claim_vec, evidence_vecs.T)[0]

        return float(np.max(similarities)) if len(similarities) > 0 else 0.0

    def _classify_claim(self, max_sim: float) -> str:
        """Classify a claim based on its maximum similarity to evidence.

        Args:
            max_sim: Maximum cosine similarity score.

        Returns:
            One of 'supported', 'refuted', or 'unverifiable'.
        """
        if max_sim >= self.sim_threshold_supported:
            return "supported"
        elif max_sim < self.sim_threshold_refuted:
            return "refuted"
        else:
            return "unverifiable"

    @staticmethod
    def _aggregate_verdict(
        verified: List[Dict[str, Any]], flagged: List[Dict[str, Any]]
    ) -> str:
        """Compute a final verdict from verified and flagged claims.

        Args:
            verified: List of verified (supported) claim dicts.
            flagged: List of flagged (refuted/unverifiable) claim dicts.

        Returns:
            Overall verdict string.
        """
        total = len(verified) + len(flagged)
        if total == 0:
            return "unverifiable"

        if not flagged:
            return "supported"

        refuted = [c for c in flagged if c["status"] == "refuted"]
        if refuted and not verified:
            return "refuted"

        return "partially_supported"

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    verifier = VerificationModule()

    answer = "Paris is the capital of France. It is located on the river Seine."
    evidence = [
        {"text": "Paris is the capital and most populous city of France.", "source": "wiki", "score": 0.9},
        {"text": "The Seine is a 777km long river in northern France.", "source": "wiki", "score": 0.7},
    ]

    print("=== Verification Module ===")
    result = verifier.verify(answer, evidence)
    print(f"  Verdict: {result['verdict']}")
    print(f"  Verified claims ({len(result['verified_claims'])}):")
    for c in result["verified_claims"]:
        print(f"    - {c['claim'][:60]}  sim={c['max_similarity']:.4f}")
    print(f"  Flagged claims ({len(result['flagged_claims'])}):")
    for c in result["flagged_claims"]:
        print(f"    - {c['claim'][:60]}  sim={c['max_similarity']:.4f}  status={c['status']}")
