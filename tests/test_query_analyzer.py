"""
Tests for modules.query_analyzer.QueryAnalyzer
===============================================

Covers:
  - Happy path with a normal factoid query
  - Complex multi-entity, multi-hop query with dates and citations
  - Empty string edge case
  - Single-word edge case
  - Type error when passing a non-string
  - Entity-type flag detection
"""

from __future__ import annotations

import pytest
from modules.query_analyzer import QueryAnalyzer


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    """Provide a QueryAnalyzer instance using default config."""
    return QueryAnalyzer()


# ------------------------------------------------------------------ #
#  Happy-path                                                         #
# ------------------------------------------------------------------ #

class TestHappyPath:
    """Basic queries that should parse cleanly."""

    def test_simple_factoid(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("What is the capital of France?")
        assert isinstance(result, dict)
        # Must have all 8 keys
        expected_keys = {
            "entity_count",
            "query_length_tokens",
            "contains_date",
            "contains_citation_pattern",
            "multi_hop_indicator",
            "entity_type_flags",
            "avg_token_length",
            "complexity_score",
        }
        assert set(result.keys()) == expected_keys

        assert result["query_length_tokens"] == 6
        assert result["contains_date"] is False
        assert result["contains_citation_pattern"] is False
        assert result["multi_hop_indicator"] is False
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_complex_multi_hop(self, analyzer: QueryAnalyzer) -> None:
        query = (
            "According to Dr. Smith at MIT, what caused the 2008 financial "
            "crisis and how did it compare to the Great Depression?"
        )
        result = analyzer.analyze(query)

        assert result["contains_date"] is True
        assert result["contains_citation_pattern"] is True
        assert result["multi_hop_indicator"] is True
        assert result["entity_count"] >= 1  # At least Dr. Smith / MIT
        assert result["complexity_score"] > 0.5  # Should be high


# ------------------------------------------------------------------ #
#  Edge cases                                                         #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    """Edge-case inputs that must not crash."""

    def test_empty_string(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("")
        assert result["query_length_tokens"] == 0
        assert result["entity_count"] == 0
        assert result["contains_date"] is False
        assert result["contains_citation_pattern"] is False
        assert result["complexity_score"] >= 0.0

    def test_single_word(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("Hello")
        assert result["query_length_tokens"] == 1
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_non_string_raises_type_error(self, analyzer: QueryAnalyzer) -> None:
        with pytest.raises(TypeError, match="query must be a str"):
            analyzer.analyze(12345)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
#  Entity type flags                                                  #
# ------------------------------------------------------------------ #

class TestEntityTypeFlags:
    """Verify that entity_type_flags detect the expected types."""

    def test_person_detected(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("Dr. Watson solved the case")
        assert result["entity_type_flags"]["PERSON"] is True

    def test_org_detected(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("Google Inc is a large company")
        assert result["entity_type_flags"]["ORG"] is True

    def test_loc_detected(self, analyzer: QueryAnalyzer) -> None:
        # Use a clear geo-political entity that both spaCy and regex fallback can detect
        result = analyzer.analyze("What is the capital of Germany?")
        assert result["entity_type_flags"]["LOC"] is True

    def test_date_flag_matches_contains_date(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("In January 2024 the event happened")
        assert result["entity_type_flags"]["DATE"] is True
        assert result["contains_date"] is True


# ------------------------------------------------------------------ #
#  Complexity score bounds                                            #
# ------------------------------------------------------------------ #

class TestComplexityBounds:
    """Complexity score must always be in [0, 1]."""

    def test_very_long_query(self, analyzer: QueryAnalyzer) -> None:
        long_query = " ".join(["word"] * 200)
        result = analyzer.analyze(long_query)
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_all_caps_query(self, analyzer: QueryAnalyzer) -> None:
        result = analyzer.analyze("THIS IS AN ALL CAPS QUERY WITH NO ENTITIES")
        assert 0.0 <= result["complexity_score"] <= 1.0
