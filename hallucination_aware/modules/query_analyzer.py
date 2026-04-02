"""
Query Analyzer Module
=====================

Analyses a raw user query and extracts a feature dictionary used downstream
by the HallucinationPredictor.

**Inputs**:  A single query string.
**Outputs**: A dictionary containing eight feature fields:
    - entity_count (int): number of named-entity-like tokens detected
    - query_length_tokens (int): token count (whitespace split)
    - contains_date (bool): whether any date/year pattern appears
    - contains_citation_pattern (bool): whether citation cues exist
    - multi_hop_indicator (bool): heuristic for multi-hop reasoning
    - entity_type_flags (Dict[str, bool]): flags for PERSON, ORG, LOC, DATE
    - avg_token_length (float): mean character length of tokens
    - complexity_score (float): 0-1 composite complexity metric

**Dependencies**: PyYAML (via load_config), re, string — no other modules.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any, Dict, List

from modules import load_config


# ---------------------------------------------------------------------------
# Regex patterns compiled once at module level
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(
    r"""
    \b\d{4}\b                              # 4-digit year
    | \b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b   # dd/mm/yyyy etc.
    | \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May
        |Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?
        |Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
      \s+\d{1,2}(?:,?\s+\d{4})?\b         # Month DD, YYYY
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(
    r"""
    according\s+to
    | cited?\s+(?:by|in|from)
    | \bsource[s]?\b
    | \breference[s]?\b
    | \[\d+\]                              # [1] style refs
    | as\s+(?:stated|reported|noted)\s+(?:by|in)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MULTI_HOP_PATTERN = re.compile(
    r"""
    \b(?:compare|contrast|difference|relation(?:ship)?|between
       |cause|effect|why\s+did|how\s+did|what\s+led
       |connect(?:ion)?|because|result(?:ed)?|impact
       |both|and\s+also|as\s+well\s+as)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Simple heuristic patterns for entity-type detection
_PERSON_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|President|King|Queen|Prince|Princess)\b\.?\s*[A-Z]",
    re.IGNORECASE,
)
_ORG_KEYWORDS = re.compile(
    r"\b(?:Inc|Corp|LLC|Ltd|Company|University|Institute|Organization|Association|Agency|Department|Ministry|Government)\b",
    re.IGNORECASE,
)
_LOC_KEYWORDS = re.compile(
    r"\b(?:City|Country|State|Province|River|Mountain|Ocean|Sea|Island|Lake|Street|Avenue|Road|Boulevard)\b",
    re.IGNORECASE,
)


def _count_capitalized_sequences(text: str) -> int:
    """Count sequences of capitalised words (rough NER proxy)."""
    tokens = text.split()
    count = 0
    i = 0
    while i < len(tokens):
        # Strip punctuation for the check
        clean = tokens[i].strip(string.punctuation)
        if clean and clean[0].isupper() and not _is_sentence_start(tokens, i):
            count += 1
            # Skip continuation of the same entity
            while (
                i + 1 < len(tokens)
                and tokens[i + 1].strip(string.punctuation)[:1].isupper()
            ):
                i += 1
        i += 1
    return count


def _is_sentence_start(tokens: List[str], idx: int) -> bool:
    """Heuristic: is this token at the start of a sentence?"""
    if idx == 0:
        return True
    prev = tokens[idx - 1]
    return prev.endswith((".","!","?"))


class QueryAnalyzer:
    """Extracts features from a raw query string for downstream prediction."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the QueryAnalyzer.

        Args:
            config: Optional pre-loaded configuration dict.
                    If None, loads from the default config path.
        """
        self.config = config if config is not None else load_config()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyse a query and return a feature dictionary.

        Args:
            query: The user query string to analyse.

        Returns:
            Dict with keys:
                entity_count, query_length_tokens, contains_date,
                contains_citation_pattern, multi_hop_indicator,
                entity_type_flags, avg_token_length, complexity_score
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        tokens = query.split()
        query_length_tokens = len(tokens)

        entity_count = _count_capitalized_sequences(query)
        contains_date = bool(_DATE_PATTERN.search(query))
        contains_citation_pattern = bool(_CITATION_PATTERN.search(query))
        multi_hop_indicator = bool(_MULTI_HOP_PATTERN.search(query))

        entity_type_flags = self._detect_entity_types(query)

        avg_token_length = (
            sum(len(t.strip(string.punctuation)) for t in tokens) / max(query_length_tokens, 1)
        )

        complexity_score = self._compute_complexity(
            entity_count=entity_count,
            query_length_tokens=query_length_tokens,
            contains_date=contains_date,
            contains_citation_pattern=contains_citation_pattern,
            multi_hop_indicator=multi_hop_indicator,
            entity_type_flags=entity_type_flags,
            avg_token_length=avg_token_length,
        )

        return {
            "entity_count": entity_count,
            "query_length_tokens": query_length_tokens,
            "contains_date": contains_date,
            "contains_citation_pattern": contains_citation_pattern,
            "multi_hop_indicator": multi_hop_indicator,
            "entity_type_flags": entity_type_flags,
            "avg_token_length": round(avg_token_length, 4),
            "complexity_score": round(complexity_score, 4),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_entity_types(query: str) -> Dict[str, bool]:
        """Return boolean flags for PERSON, ORG, LOC, DATE entity types."""
        return {
            "PERSON": bool(_PERSON_PATTERN.search(query)),
            "ORG": bool(_ORG_KEYWORDS.search(query)),
            "LOC": bool(_LOC_KEYWORDS.search(query)),
            "DATE": bool(_DATE_PATTERN.search(query)),
        }

    @staticmethod
    def _compute_complexity(
        entity_count: int,
        query_length_tokens: int,
        contains_date: bool,
        contains_citation_pattern: bool,
        multi_hop_indicator: bool,
        entity_type_flags: Dict[str, bool],
        avg_token_length: float,
    ) -> float:
        """Compute a 0-1 composite complexity score.

        Combines several signals with hand-tuned weights.  The score
        is pushed through a sigmoid so it stays in [0, 1].
        """
        raw = 0.0
        # Entity density — more entities ⇒ harder
        raw += min(entity_count, 5) * 0.12
        # Length — longer queries tend to be more complex
        raw += min(query_length_tokens, 40) * 0.01
        # Boolean flags
        raw += 0.15 if contains_date else 0.0
        raw += 0.20 if contains_citation_pattern else 0.0
        raw += 0.25 if multi_hop_indicator else 0.0
        # Entity type diversity
        type_count = sum(entity_type_flags.values())
        raw += type_count * 0.08
        # Avg token length (longer words ⇒ more technical)
        raw += min(avg_token_length, 10) * 0.02

        # Sigmoid normalisation to [0, 1]
        score = 1.0 / (1.0 + math.exp(-( raw - 0.6) * 4))
        return score


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    test_queries = [
        "What is the capital of France?",
        "According to Dr. Smith at MIT, what caused the 2008 financial crisis and how did it compare to the Great Depression?",
        "Hello",
        "",
        "When was President Lincoln born and what was the relationship between his policies and the Civil War outcome in 1865?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q!r}")
        features = analyzer.analyze(q)
        for k, v in features.items():
            print(f"  {k}: {v}")
