"""
Tests for modules.generation_module.GenerationModule
=====================================================

Covers:
  - Happy path for each strategy (direct_llm, rag, rag_verification)
  - Returned dict structure
  - Edge cases: empty query, invalid strategy, non-string input
  - Token counting sanity
"""

from __future__ import annotations

import pytest
from modules.generation_module import GenerationModule


@pytest.fixture
def gen() -> GenerationModule:
    return GenerationModule()


# ------------------------------------------------------------------ #
#  Happy-path                                                          #
# ------------------------------------------------------------------ #

class TestHappyPath:
    def test_direct_llm(self, gen: GenerationModule) -> None:
        result = gen.generate("What is 2+2?", "direct_llm")
        assert set(result.keys()) == {"answer", "strategy_used", "prompt_tokens"}
        assert result["strategy_used"] == "direct_llm"
        assert isinstance(result["answer"], str)
        assert result["prompt_tokens"] > 0

    def test_rag_with_context(self, gen: GenerationModule) -> None:
        ctx = ["The answer to 2+2 is 4."]
        result = gen.generate("What is 2+2?", "rag", context=ctx)
        assert result["strategy_used"] == "rag"
        assert result["prompt_tokens"] > 0

    def test_rag_verification(self, gen: GenerationModule) -> None:
        ctx = ["The answer to 2+2 is 4."]
        result = gen.generate("What is 2+2?", "rag_verification", context=ctx)
        assert result["strategy_used"] == "rag_verification"

    def test_rag_without_context(self, gen: GenerationModule) -> None:
        result = gen.generate("What is 2+2?", "rag")
        assert result["strategy_used"] == "rag"
        assert "answer" in result


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_query(self, gen: GenerationModule) -> None:
        result = gen.generate("", "direct_llm")
        assert isinstance(result["answer"], str)

    def test_invalid_strategy_raises(self, gen: GenerationModule) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            gen.generate("Hello", "invalid_strategy")

    def test_non_string_query_raises(self, gen: GenerationModule) -> None:
        with pytest.raises(TypeError, match="query must be a str"):
            gen.generate(123, "direct_llm")  # type: ignore[arg-type]

    def test_prompt_tokens_increase_with_context(self, gen: GenerationModule) -> None:
        r1 = gen.generate("Question?", "rag", context=[])
        r2 = gen.generate("Question?", "rag", context=["A long document. " * 50])
        assert r2["prompt_tokens"] > r1["prompt_tokens"]
