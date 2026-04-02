"""
Generation Module
=================

Wraps LLM generation (OpenAI chat API) and constructs prompts based on the
selected strategy.

**Inputs**:  Query string, strategy name, optional context docs.
**Outputs**: ``{ answer: str, strategy_used: str, prompt_tokens: int }``
**Dependencies**: openai, tiktoken, PyYAML — no other project modules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import tiktoken

from modules import load_config

# Lazy import openai so the module can be tested without an API key
_openai = None


def _get_openai():
    """Lazy-load the openai module."""
    global _openai
    if _openai is None:
        import openai
        _openai = openai
    return _openai


class GenerationModule:
    """Generate answers using an LLM with strategy-dependent prompts."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the GenerationModule.

        Args:
            config: Pre-loaded config dict. If None, loads default.
        """
        self.config = config if config is not None else load_config()
        model_cfg = self.config.get("model", {})

        self.model_name: str = model_cfg.get("name", "gpt-3.5-turbo")
        self.temperature: float = float(model_cfg.get("temperature", 0.0))
        self.max_tokens: int = int(model_cfg.get("max_tokens", 256))

        # Token counter
        try:
            self._enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        strategy: str,
        context: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Generate an answer for *query* using the given *strategy*.

        Args:
            query: The user question.
            strategy: One of ``'direct_llm'``, ``'rag'``, ``'rag_verification'``.
            context: Optional list of retrieved text passages (used by
                     ``'rag'`` and ``'rag_verification'`` strategies).

        Returns:
            Dict with ``answer``, ``strategy_used``, and ``prompt_tokens``.

        Raises:
            TypeError: If *query* is not a str.
            ValueError: If *strategy* is not a recognised value.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        valid_strategies = {"direct_llm", "rag", "rag_verification"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {valid_strategies}"
            )

        if context is None:
            context = []

        messages = self._build_messages(query, strategy, context)
        prompt_text = " ".join(m["content"] for m in messages)
        prompt_tokens = len(self._enc.encode(prompt_text))

        # Call the LLM
        answer = self._call_llm(messages)

        return {
            "answer": answer,
            "strategy_used": strategy,
            "prompt_tokens": prompt_tokens,
        }

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self, query: str, strategy: str, context: List[str]
    ) -> List[Dict[str, str]]:
        """Build the chat-style message list for the LLM.

        Returns:
            List of ``{role, content}`` dicts.
        """
        if strategy == "direct_llm":
            system_msg = (
                "You are a helpful and accurate assistant. Answer the "
                "following question concisely and truthfully."
            )
            return [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ]

        # RAG / RAG+Verification share the same prompt structure
        context_block = "\n\n".join(
            f"[Document {i+1}]: {doc}" for i, doc in enumerate(context)
        )
        if not context_block:
            context_block = "(No supporting documents retrieved.)"

        if strategy == "rag_verification":
            system_msg = (
                "You are a careful, fact-checking assistant. Use ONLY the "
                "provided documents to answer the question. If the documents "
                "do not contain enough information, say so explicitly. "
                "Cite document numbers where applicable."
            )
        else:  # "rag"
            system_msg = (
                "You are a helpful assistant. Use the provided documents to "
                "answer the question. Prefer information from the documents "
                "over your own knowledge."
            )

        user_msg = (
            f"Documents:\n{context_block}\n\n"
            f"Question: {query}"
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------ #
    #  LLM call                                                           #
    # ------------------------------------------------------------------ #

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI Chat API.

        Falls back to a placeholder response if no API key is set, so that
        downstream modules and tests remain functional.

        Args:
            messages: Chat messages list.

        Returns:
            The assistant's response text.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Graceful fallback for testing / offline usage
            return (
                "[LLM response placeholder — set OPENAI_API_KEY to enable "
                "real generation]"
            )

        try:
            openai = _get_openai()
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return f"[LLM call failed: {exc}]"


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    gen = GenerationModule()

    print("=== direct_llm strategy ===")
    r1 = gen.generate("What is the capital of France?", "direct_llm")
    for k, v in r1.items():
        print(f"  {k}: {v}")

    print("\n=== rag strategy (with context) ===")
    ctx = ["Paris is the capital and largest city of France."]
    r2 = gen.generate("What is the capital of France?", "rag", context=ctx)
    for k, v in r2.items():
        print(f"  {k}: {v}")

    print("\n=== rag_verification strategy ===")
    r3 = gen.generate("Who founded Tesla?", "rag_verification", context=ctx)
    for k, v in r3.items():
        print(f"  {k}: {v}")
