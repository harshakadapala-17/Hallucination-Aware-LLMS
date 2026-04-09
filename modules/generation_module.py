"""
Generation Module
=================

Wraps LLM generation with support for multiple backends:
  - OpenAI (default): uses the OpenAI chat API (requires OPENAI_API_KEY).
  - Ollama (local): calls a locally running Ollama server (no API key needed).
    Set model.provider: "ollama" and model.name: "llama3" (or any Ollama model)
    in config.yaml, then run `ollama serve` before using the system.

Falls back to a placeholder response if no backend is available.

**Inputs**:  Query string, strategy name, optional context docs.
**Outputs**: ``{ answer: str, strategy_used: str, prompt_tokens: int }``
**Dependencies**: openai, tiktoken, PyYAML — no other project modules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

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
        # "openai" (default) or "ollama"
        self.provider: str = model_cfg.get("provider", "openai")
        self.ollama_base_url: str = model_cfg.get("ollama_base_url", "http://localhost:11434")

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
        retry_hint: str | None = None,
    ) -> Dict[str, Any]:
        """Generate an answer for *query* using the given *strategy*.

        Args:
            query: The user question.
            strategy: One of ``'direct_llm'``, ``'rag'``, ``'rag_verification'``.
            context: Optional list of retrieved text passages.
            retry_hint: Optional instruction prepended to the system message
                        on a re-generation attempt (e.g. after a refuted verdict).

        Returns:
            Dict with ``answer``, ``strategy_used``, and ``prompt_tokens``.
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

        messages = self._build_messages(query, strategy, context, retry_hint=retry_hint)
        prompt_text = " ".join(m["content"] for m in messages)
        prompt_tokens = len(self._enc.encode(prompt_text))

        answer = self._call_llm(messages)

        return {
            "answer": answer,
            "strategy_used": strategy,
            "prompt_tokens": prompt_tokens,
        }

    def generate_stream(
        self,
        query: str,
        strategy: str,
        context: List[str] | None = None,
    ):
        """Stream the LLM response token-by-token.

        Yields string chunks as they arrive. Falls back to yielding the full
        response at once if streaming is not supported by the backend.

        Usage (Streamlit example):
            with st.empty() as placeholder:
                full = ""
                for chunk in gen.generate_stream(query, strategy, context):
                    full += chunk
                    placeholder.markdown(full)

        Args:
            query: The user question.
            strategy: One of ``'direct_llm'``, ``'rag'``, ``'rag_verification'``.
            context: Optional retrieved text passages.

        Yields:
            str chunks of the assistant's response.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        valid_strategies = {"direct_llm", "rag", "rag_verification"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy '{strategy}'.")

        if context is None:
            context = []

        messages = self._build_messages(query, strategy, context)
        yield from self._stream_llm(messages)

    def _stream_llm(self, messages: List[Dict[str, str]]):
        """Yield response chunks from the configured backend."""
        if self.provider == "ollama":
            yield from self._stream_ollama(messages)
        else:
            yield from self._stream_openai(messages)

    def _stream_openai(self, messages: List[Dict[str, str]]):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            yield "[LLM response placeholder — set OPENAI_API_KEY to enable real generation]"
            return
        try:
            openai = _get_openai()
            client = openai.OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"[OpenAI stream failed: {exc}]"

    def _stream_ollama(self, messages: List[Dict[str, str]]):
        try:
            openai = _get_openai()
            client = openai.OpenAI(
                api_key="ollama",
                base_url=f"{self.ollama_base_url}/v1",
            )
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"[Ollama stream failed: {exc}]"

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        query: str,
        strategy: str,
        context: List[str],
        retry_hint: str | None = None,
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
            if retry_hint:
                system_msg = f"{retry_hint}\n\n{system_msg}"
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

        if retry_hint:
            system_msg = f"{retry_hint}\n\n{system_msg}"

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
        """Dispatch to the configured LLM backend (OpenAI or Ollama).

        Returns a placeholder string if no backend is available so that
        downstream modules and tests remain functional.

        Args:
            messages: Chat messages list.

        Returns:
            The assistant's response text.
        """
        if self.provider == "ollama":
            return self._call_ollama(messages)
        return self._call_openai(messages)

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI Chat Completions API."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return (
                "[LLM response placeholder — set OPENAI_API_KEY or switch to "
                "provider: ollama in config.yaml]"
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
            return f"[OpenAI call failed: {exc}]"

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call a locally running Ollama server.

        Requires `ollama serve` to be running. Install Ollama from ollama.com,
        then pull a model: `ollama pull llama3`.

        Uses the OpenAI-compatible REST endpoint that Ollama exposes at
        /v1/chat/completions so no extra SDK is needed.
        """
        try:
            openai = _get_openai()
            client = openai.OpenAI(
                api_key="ollama",  # Ollama doesn't check the key value
                base_url=f"{self.ollama_base_url}/v1",
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return (
                f"[Ollama call failed: {exc}. "
                f"Is `ollama serve` running at {self.ollama_base_url}?]"
            )


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
