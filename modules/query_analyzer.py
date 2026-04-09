"""
Query Analyzer Module
=====================

Analyses a raw user query and extracts a feature dictionary used downstream
by the HallucinationPredictor.

**Inputs**:  A single query string.
**Outputs**: A dictionary containing eight feature fields:
    - entity_count (int): number of named entities detected by spaCy
    - query_length_tokens (int): token count (whitespace split)
    - contains_date (bool): whether any date/year pattern appears
    - contains_citation_pattern (bool): whether citation cues exist
    - multi_hop_indicator (bool): heuristic for multi-hop reasoning
    - entity_type_flags (Dict[str, bool]): flags for PERSON, ORG, LOC, DATE
    - avg_token_length (float): mean character length of tokens
    - complexity_score (float): 0-1 composite complexity metric

**Dependencies**: spaCy (en_core_web_sm), PyYAML, re, string.
Falls back to regex-based NER if spaCy model is not available.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any, Dict, List

from modules import load_config

# ---------------------------------------------------------------------------
# spaCy — lazy-loaded so import errors don't crash the whole module
# ---------------------------------------------------------------------------

_nlp = None
_spacy_available = False


def _get_nlp():
    """Lazy-load the spaCy pipeline (en_core_web_sm)."""
    global _nlp, _spacy_available
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        _spacy_available = True
    except (ImportError, OSError):
        _nlp = None
        _spacy_available = False
    return _nlp


# ---------------------------------------------------------------------------
# Regex patterns — used as fallback and for citation/date/multi-hop detection
# (spaCy doesn't do these well)
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(
    r"""
    \b\d{4}\b
    | \b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b
    | \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May
        |Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?
        |Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
      \s+\d{1,2}(?:,?\s+\d{4})?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(
    r"""
    according\s+to
    | cited?\s+(?:by|in|from)
    | \bsource[s]?\b
    | \breference[s]?\b
    | \[\d+\]
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

# Fallback regex patterns (used when spaCy is unavailable)
_PERSON_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|President|King|Queen|Prince|Princess)\b\.?\s*[A-Z]",
    re.IGNORECASE,
)
_ORG_KEYWORDS = re.compile(
    r"\b(?:Inc|Corp|LLC|Ltd|Company|University|Institute|Organization|"
    r"Association|Agency|Department|Ministry|Government)\b",
    re.IGNORECASE,
)
_LOC_KEYWORDS = re.compile(
    r"\b(?:City|Country|State|Province|River|Mountain|Ocean|Sea|Island|"
    r"Lake|Street|Avenue|Road|Boulevard|Capital|Region|Territory|Nation)\b",
    re.IGNORECASE,
)


def _count_capitalized_sequences(text: str) -> int:
    """Regex-based entity count fallback."""
    tokens = text.split()
    count = 0
    i = 0
    while i < len(tokens):
        clean = tokens[i].strip(string.punctuation)
        if clean and clean[0].isupper() and not _is_sentence_start(tokens, i):
            count += 1
            while (
                i + 1 < len(tokens)
                and tokens[i + 1].strip(string.punctuation)[:1].isupper()
            ):
                i += 1
        i += 1
    return count


def _is_sentence_start(tokens: List[str], idx: int) -> bool:
    if idx == 0:
        return True
    return tokens[idx - 1].endswith((".", "!", "?"))


class QueryAnalyzer:
    """Extracts features from a raw query string for downstream prediction.

    Uses spaCy en_core_web_sm for named entity recognition when available,
    falling back to regex-based heuristics otherwise.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        # Eagerly attempt to load spaCy so the first call isn't slow
        _get_nlp()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyse a query and return a feature dictionary.

        Args:
            query: The user query string.

        Returns:
            Dict with keys: entity_count, query_length_tokens, contains_date,
            contains_citation_pattern, multi_hop_indicator, entity_type_flags,
            avg_token_length, complexity_score.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        tokens = query.split()
        query_length_tokens = len(tokens)
        avg_token_length = (
            sum(len(t.strip(string.punctuation)) for t in tokens)
            / max(query_length_tokens, 1)
        )

        contains_date = bool(_DATE_PATTERN.search(query))
        contains_citation_pattern = bool(_CITATION_PATTERN.search(query))
        multi_hop_indicator = bool(_MULTI_HOP_PATTERN.search(query))

        # Entity detection: prefer spaCy, fall back to regex
        nlp = _get_nlp()
        if nlp is not None:
            entity_count, entity_type_flags = self._spacy_entities(query, nlp)
            # Override DATE flag from spaCy with regex (more reliable for date formats)
            entity_type_flags["DATE"] = contains_date
        else:
            entity_count = _count_capitalized_sequences(query)
            entity_type_flags = self._regex_entity_types(query)

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
    def _spacy_entities(query: str, nlp: Any) -> tuple[int, Dict[str, bool]]:
        """Run spaCy NER and return (entity_count, entity_type_flags)."""
        doc = nlp(query)
        entity_count = len(doc.ents)

        # Map spaCy labels to our 4 categories
        # PERSON → PERSON
        # ORG → ORG
        # GPE (geo-political), LOC, FAC → LOC
        # DATE, TIME → DATE (also covered by regex)
        flags: Dict[str, bool] = {"PERSON": False, "ORG": False, "LOC": False, "DATE": False}
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                flags["PERSON"] = True
            elif ent.label_ == "ORG":
                flags["ORG"] = True
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                flags["LOC"] = True
            elif ent.label_ in ("DATE", "TIME"):
                flags["DATE"] = True

        return entity_count, flags

    @staticmethod
    def _regex_entity_types(query: str) -> Dict[str, bool]:
        """Fallback: detect entity types via regex."""
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
        """Compute a 0-1 composite complexity score via sigmoid normalisation."""
        raw = 0.0
        raw += min(entity_count, 5) * 0.12
        raw += min(query_length_tokens, 40) * 0.01
        raw += 0.15 if contains_date else 0.0
        raw += 0.20 if contains_citation_pattern else 0.0
        raw += 0.25 if multi_hop_indicator else 0.0
        raw += sum(entity_type_flags.values()) * 0.08
        raw += min(avg_token_length, 10) * 0.02
        return 1.0 / (1.0 + math.exp(-(raw - 0.6) * 4))


# ----------------------------------------------------------------------- #
#  Smoke-test entry point                                                  #
# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    test_queries = [
        "What is the capital of France?",
        "According to Dr. Smith at MIT, what caused the 2008 financial crisis?",
        "",
        "When was President Lincoln born and how did his policies affect the Civil War?",
    ]
    spacy_status = "spaCy" if _spacy_available else "regex fallback"
    print(f"Entity detection: {spacy_status}\n")
    for q in test_queries:
        print(f"Query: {q!r}")
        features = analyzer.analyze(q)
        for k, v in features.items():
            print(f"  {k}: {v}")
        print()
