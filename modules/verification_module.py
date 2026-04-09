"""
Verification Module
===================

Post-generation fact-checking that compares an LLM answer against retrieved
evidence to flag unsupported or refuted claims.

Two verification modes (controlled by config):
  - NLI mode (default when use_nli_entailment: true): uses a cross-encoder
    NLI model (cross-encoder/nli-deberta-v3-small) to classify each claim as
    entailment / contradiction / neutral against evidence.
  - Cosine similarity mode (fallback): uses sentence-transformer embeddings.

**Inputs**:  Answer string + list of evidence dicts from RAGModule.
**Outputs**: ``{ verdict: str, verified_claims: List, flagged_claims: List }``
**Dependencies**: transformers (NLI), sentence-transformers (cosine fallback),
                  torch, numpy, PyYAML.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from modules import load_config

# ---------------------------------------------------------------------------
# Lazy-loaded backends
# ---------------------------------------------------------------------------

_sentence_transformer_cls = None
_nli_tokenizer = None
_nli_model = None
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def _get_sentence_transformer():
    global _sentence_transformer_cls
    if _sentence_transformer_cls is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_cls = SentenceTransformer
    return _sentence_transformer_cls


def _get_nli_model():
    """Lazy-load the NLI cross-encoder model."""
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(
                _NLI_MODEL_NAME
            )
            _nli_model.eval()
        except Exception:
            _nli_model = None
            _nli_tokenizer = None
    return _nli_tokenizer, _nli_model


class VerificationModule:
    """Verify an LLM-generated answer against retrieved evidence.

    When ``use_nli_entailment`` is True (default), uses a cross-encoder NLI
    model to determine entailment/contradiction for each claim.
    Falls back to cosine similarity if the NLI model cannot be loaded.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        ver_cfg = self.config.get("verification", {})

        self.sim_threshold_supported: float = float(
            ver_cfg.get("sim_threshold_supported", 0.65)
        )
        self.sim_threshold_refuted: float = float(
            ver_cfg.get("sim_threshold_refuted", 0.40)
        )
        self.max_claims: int = int(ver_cfg.get("max_claims", 5))
        # NLI entailment is now the default; set to false to use cosine sim
        self.use_nli_entailment: bool = bool(
            ver_cfg.get("use_nli_entailment", True)
        )

        rag_cfg = self.config.get("rag", {})
        self._embedding_model_name: str = rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._embedding_model = None  # lazy-loaded for cosine fallback

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def verify(
        self, answer: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify an answer against evidence documents.

        Args:
            answer: The LLM-generated answer text.
            evidence: List of ``{text, source, score}`` dicts from RAGModule.

        Returns:
            Dict with ``verdict``, ``verified_claims``, ``flagged_claims``.
            Each claim entry also carries ``method`` ('nli' or 'cosine').
        """
        if not isinstance(answer, str):
            raise TypeError(f"answer must be a str, got {type(answer).__name__}")
        if not isinstance(evidence, list):
            raise TypeError(f"evidence must be a list, got {type(evidence).__name__}")

        claims = self._extract_claims(answer)

        if not claims:
            return {"verdict": "unverifiable", "verified_claims": [], "flagged_claims": []}

        if not evidence:
            return {
                "verdict": "unverifiable",
                "verified_claims": [],
                "flagged_claims": [
                    {"claim": c, "status": "unverifiable", "score": 0.0, "method": "none"}
                    for c in claims
                ],
            }

        evidence_texts = [e.get("text", "") for e in evidence]
        verified_claims: List[Dict[str, Any]] = []
        flagged_claims: List[Dict[str, Any]] = []

        # Choose verification method
        use_nli = self.use_nli_entailment
        if use_nli:
            tok, mdl = _get_nli_model()
            use_nli = tok is not None and mdl is not None

        for claim in claims:
            if use_nli:
                status, score = self._nli_classify(claim, evidence_texts)
                method = "nli"
            else:
                score = self._max_cosine_similarity(claim, evidence_texts)
                status = self._cosine_classify(score)
                method = "cosine"

            entry: Dict[str, Any] = {
                "claim": claim,
                "status": status,
                "score": round(score, 4),
                "method": method,
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
    #  NLI verification                                                   #
    # ------------------------------------------------------------------ #

    def _nli_classify(
        self, claim: str, evidence_texts: List[str]
    ) -> Tuple[str, float]:
        """Classify a claim against evidence using NLI cross-encoder.

        Runs claim against each evidence chunk. Takes the max entailment
        score across chunks; uses max contradiction score for refuted.

        Returns:
            (status, score) where score is the max entailment probability.
        """
        import torch

        tok, model = _get_nli_model()
        id2label: Dict[int, str] = model.config.id2label  # e.g. {0:'contradiction', 1:'entailment', 2:'neutral'}

        max_entail = 0.0
        max_contradict = 0.0

        for evidence_text in evidence_texts:
            if not evidence_text.strip():
                continue
            # Premise = evidence, hypothesis = claim
            inputs = tok(
                evidence_text,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

            for idx, label in id2label.items():
                p = float(probs[idx])
                if "entail" in label.lower():
                    max_entail = max(max_entail, p)
                elif "contradict" in label.lower():
                    max_contradict = max(max_contradict, p)

        # Decision: entailment wins if clearly dominant
        if max_entail >= 0.6:
            return "supported", max_entail
        elif max_contradict >= 0.5:
            return "refuted", max_contradict
        else:
            return "unverifiable", max(max_entail, max_contradict)

    # ------------------------------------------------------------------ #
    #  Cosine similarity fallback                                          #
    # ------------------------------------------------------------------ #

    def _max_cosine_similarity(self, claim: str, evidence_texts: List[str]) -> float:
        model = self._get_embedding_model()
        all_texts = [claim] + evidence_texts
        embeddings = np.array(
            model.encode(all_texts, normalize_embeddings=True), dtype=np.float32
        )
        similarities = np.dot(embeddings[0:1], embeddings[1:].T)[0]
        return float(np.max(similarities)) if len(similarities) > 0 else 0.0

    def _cosine_classify(self, max_sim: float) -> str:
        if max_sim >= self.sim_threshold_supported:
            return "supported"
        elif max_sim < self.sim_threshold_refuted:
            return "refuted"
        return "unverifiable"

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_claims(self, answer: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5][: self.max_claims]

    @staticmethod
    def _aggregate_verdict(
        verified: List[Dict[str, Any]], flagged: List[Dict[str, Any]]
    ) -> str:
        total = len(verified) + len(flagged)
        if total == 0:
            return "unverifiable"
        if not flagged:
            return "supported"
        refuted = [c for c in flagged if c["status"] == "refuted"]
        if refuted and not verified:
            return "refuted"
        return "partially_supported"

    def _get_embedding_model(self):
        if self._embedding_model is None:
            SentenceTransformer = _get_sentence_transformer()
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model


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

    result = verifier.verify(answer, evidence)
    print(f"Verdict: {result['verdict']}")
    for c in result["verified_claims"]:
        print(f"  [SUPPORTED/{c['method']}] {c['claim'][:60]}")
    for c in result["flagged_claims"]:
        print(f"  [{c['status'].upper()}/{c['method']}] {c['claim'][:60]}")
