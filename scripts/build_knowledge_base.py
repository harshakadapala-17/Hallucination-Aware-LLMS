"""
Build Knowledge Base Script
============================

Fetches real Wikipedia articles via the Wikipedia REST API and saves them
as a JSONL knowledge base that can be indexed by build_index.py.

Uses the Wikipedia REST API directly (no `wikipedia` package needed).
Requires only `requests`, which is already a transitive dependency.

Usage:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --output data/knowledge_base.jsonl
    python scripts/build_knowledge_base.py --max-sections 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Topics — primary article titles to fetch
# ---------------------------------------------------------------------------

TOPICS = [
    # Inventions / inventors
    "Thomas Edison",
    "Nikola Tesla",
    "Incandescent light bulb",
    "Alexander Graham Bell",
    "Telephone",
    # Finance / economics
    "2008 financial crisis",
    "Great Depression",
    "Wall Street Crash of 1929",
    "Gross domestic product",
    "World Bank",
    # Science
    "Photosynthesis",
    "Chlorophyll",
    "Albert Einstein",
    "Theory of relativity",
    "Quantum mechanics",
    "DNA",
    "Genetics",
    "Charles Darwin",
    "Evolution",
    # Technology / CS
    "Artificial intelligence",
    "Machine learning",
    "Neural network",
    "Python (programming language)",
    "Computer science",
    "Alan Turing",
    # Geography
    "France",
    "Paris",
    "Eiffel Tower",
    "Japan",
    "Germany",
    "Berlin",
    "Tokyo",
    # Climate
    "Climate change",
    "Global warming",
    "Greenhouse effect",
    # Health
    "COVID-19",
    "Pandemic",
    "Vaccine",
    # Space
    "Solar System",
    "Moon landing",
    "NASA",
    # History / WWII
    "World War II",
    "Adolf Hitler",
    "Winston Churchill",
    # Literature
    "William Shakespeare",
    "Romeo and Juliet",
]

# Wikipedia REST API base URL
_API_BASE = "https://en.wikipedia.org/api/rest_v1"
_MEDIAWIKI_BASE = "https://en.wikipedia.org/w/api.php"
_HEADERS = {
    "User-Agent": (
        "HallucinationAwareLLM/1.0 "
        "(Academic research project; uses Wikipedia content for RAG; "
        "https://github.com/example/hallucination-aware-llms)"
    )
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove Wikipedia markup artifacts and normalise whitespace.

    Args:
        text: Raw text from Wikipedia API.

    Returns:
        Cleaned plain-text string.
    """
    # Remove citation markers like [1], [2], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove section headings (== Heading ==)
    text = re.sub(r"={2,}[^=]+=={2,}", "", text)
    # Collapse excessive newlines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-level chunks.

    Args:
        text: Cleaned article text.
        chunk_size: Target words per chunk.
        overlap: Overlapping words between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 15) -> dict | None:
    """Make a GET request and return parsed JSON or None on failure."""
    try:
        import requests
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_summary(title: str) -> str:
    """Fetch the Wikipedia REST API summary for a page title.

    Args:
        title: Wikipedia article title (URL-encoded automatically).

    Returns:
        Extract text or empty string on failure.
    """
    safe_title = title.replace(" ", "_")
    data = _get(f"{_API_BASE}/page/summary/{safe_title}")
    if data:
        return data.get("extract", "")
    return ""


def fetch_full_text(title: str, max_sections: int = 5) -> str:
    """Fetch full article text via the MediaWiki API.

    Args:
        title: Wikipedia article title.
        max_sections: Approximate number of top-level sections to include.

    Returns:
        Plain-text content or empty string on failure.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "format": "json",
        "redirects": "1",
    }
    data = _get(_MEDIAWIKI_BASE, params=params)
    if not data:
        return ""

    try:
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "") or ""

        # Keep only the first max_sections * ~3 paragraphs of content
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        selected = paragraphs[: max_sections * 3]
        return "\n\n".join(selected)
    except Exception:
        return ""


def search_wikipedia(query: str, n: int = 3) -> list[str]:
    """Search Wikipedia and return the top-n article titles.

    Args:
        query: Search term.
        n: Number of results to return.

    Returns:
        List of article title strings.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": n,
        "namespace": "0",
        "format": "json",
    }
    data = _get(_MEDIAWIKI_BASE, params=params)
    if data and len(data) >= 2:
        return list(data[1])
    return []


# ---------------------------------------------------------------------------
# Per-article fetching
# ---------------------------------------------------------------------------

def fetch_article(title: str, max_sections: int) -> list[dict]:
    """Fetch one Wikipedia article and return document chunks.

    Args:
        title: Article title.
        max_sections: Max content sections to include.

    Returns:
        List of ``{"text": str, "source": str}`` dicts.
    """
    source = f"wikipedia:{title}"
    docs: list[dict] = []

    # Summary
    summary = clean_text(fetch_summary(title))
    if summary:
        for chunk in split_into_chunks(summary, chunk_size=300):
            docs.append({"text": chunk, "source": source})

    # Full content
    full = clean_text(fetch_full_text(title, max_sections))
    if full:
        for chunk in split_into_chunks(full, chunk_size=400, overlap=80):
            # Skip if it duplicates the summary exactly
            if chunk.strip() and chunk not in summary:
                docs.append({"text": chunk, "source": source})

    return docs


def fetch_topic(
    topic: str,
    max_sections: int,
    fetched_titles: set,
) -> list[dict]:
    """Try to fetch the topic by exact title, then via search.

    Args:
        topic: Primary search term / article title.
        max_sections: Passed to fetch_article.
        fetched_titles: Set of already-fetched titles (modified in place).

    Returns:
        List of document dicts.
    """
    # Try exact title first
    if topic not in fetched_titles:
        docs = fetch_article(topic, max_sections)
        if docs:
            fetched_titles.add(topic)
            return docs

    # Fall back to Wikipedia search
    candidates = search_wikipedia(topic, n=3)
    for candidate in candidates:
        if candidate not in fetched_titles:
            docs = fetch_article(candidate, max_sections)
            if docs:
                fetched_titles.add(candidate)
                return docs

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia articles and save to a JSONL knowledge base."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/knowledge_base.jsonl",
        help="Output JSONL file path (default: data/knowledge_base.jsonl).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=5,
        help="Maximum content sections per article (default: 5).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to wait between requests (default: 0.3).",
    )
    args = parser.parse_args()

    # Verify requests is available
    try:
        import requests  # noqa: F401
    except ImportError:
        print("ERROR: 'requests' package not installed. Run: pip install requests")
        sys.exit(1)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(TOPICS)} topics from Wikipedia REST API ...")
    print(f"Output: {output_path}")
    print("=" * 60)

    all_docs: list[dict] = []
    fetched_titles: set = set()
    ok, skipped = 0, 0

    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i:>2}/{len(TOPICS)}] {topic} ... ", end="", flush=True)

        docs = fetch_topic(topic, args.max_sections, fetched_titles)

        if docs:
            all_docs.extend(docs)
            ok += 1
            print(f"OK ({len(docs)} chunks)")
        else:
            skipped += 1
            print("SKIP")

        if i < len(TOPICS):
            time.sleep(args.delay)

    print("=" * 60)
    print(f"Articles fetched:  {ok}")
    print(f"Articles skipped:  {skipped}")
    print(f"Total chunks:      {len(all_docs)}")

    if not all_docs:
        print(
            "ERROR: No documents fetched. "
            "Check your internet connection and that Wikipedia is reachable."
        )
        sys.exit(1)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved to:          {output_path}  ({size_kb:.1f} KB)")
    print("Done. Run 'python scripts/build_index.py' to index the knowledge base.")


if __name__ == "__main__":
    main()
