"""
Tests for modules.verification_module.VerificationModule
=========================================================

Covers:
  - Happy path: answer supported by evidence
  - No evidence → unverifiable
  - Empty answer → unverifiable
  - Non-string / non-list type errors
  - Claim extraction edge cases
"""

from __future__ import annotations

import pytest
from modules.verification_module import VerificationModule


@pytest.fixture
def verifier() -> VerificationModule:
    return VerificationModule()


GOOD_EVIDENCE = [
    {"text": "Paris is the capital and most populous city of France.", "source": "wiki", "score": 0.9},
    {"text": "The Seine is a long river flowing through Paris, France.", "source": "wiki", "score": 0.8},
]


# ------------------------------------------------------------------ #
#  Happy-path                                                          #
# ------------------------------------------------------------------ #

class TestHappyPath:
    def test_supported_answer(self, verifier: VerificationModule) -> None:
        answer = "Paris is the capital of France."
        result = verifier.verify(answer, GOOD_EVIDENCE)
        assert set(result.keys()) == {"verdict", "verified_claims", "flagged_claims"}
        assert result["verdict"] in {"supported", "partially_supported"}
        assert isinstance(result["verified_claims"], list)
        assert isinstance(result["flagged_claims"], list)

    def test_claim_entries_have_required_keys(self, verifier: VerificationModule) -> None:
        answer = "Paris is the capital of France."
        result = verifier.verify(answer, GOOD_EVIDENCE)
        all_claims = result["verified_claims"] + result["flagged_claims"]
        for claim in all_claims:
            assert "claim" in claim
            assert "status" in claim
            assert "max_similarity" in claim


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_no_evidence(self, verifier: VerificationModule) -> None:
        result = verifier.verify("Paris is the capital.", [])
        assert result["verdict"] == "unverifiable"
        assert len(result["flagged_claims"]) > 0

    def test_empty_answer(self, verifier: VerificationModule) -> None:
        result = verifier.verify("", GOOD_EVIDENCE)
        assert result["verdict"] == "unverifiable"
        assert result["verified_claims"] == []
        assert result["flagged_claims"] == []

    def test_short_answer(self, verifier: VerificationModule) -> None:
        # Very short string (< 5 chars) should be filtered out by claim extractor
        result = verifier.verify("Yes.", GOOD_EVIDENCE)
        assert result["verdict"] == "unverifiable"

    def test_non_string_answer_raises(self, verifier: VerificationModule) -> None:
        with pytest.raises(TypeError, match="answer must be a str"):
            verifier.verify(123, GOOD_EVIDENCE)  # type: ignore[arg-type]

    def test_non_list_evidence_raises(self, verifier: VerificationModule) -> None:
        with pytest.raises(TypeError, match="evidence must be a list"):
            verifier.verify("answer", "not a list")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
#  Claim extraction                                                    #
# ------------------------------------------------------------------ #

class TestClaimExtraction:
    def test_multi_sentence(self, verifier: VerificationModule) -> None:
        answer = "Paris is the capital. Berlin is in Germany. Tokyo is in Japan."
        claims = verifier._extract_claims(answer)
        assert len(claims) == 3

    def test_max_claims_limit(self, verifier: VerificationModule) -> None:
        answer = ". ".join([f"Sentence number {i}" for i in range(20)]) + "."
        claims = verifier._extract_claims(answer)
        assert len(claims) <= verifier.max_claims

    def test_empty_sentences_filtered(self, verifier: VerificationModule) -> None:
        answer = "   .  .  Real claim here."
        claims = verifier._extract_claims(answer)
        # Only the real claim should survive (len > 5)
        assert all(len(c) > 5 for c in claims)
