"""
Build Index Script
==================

Loads documents from a JSONL file and builds a FAISS index using RAGModule.

Default behaviour (no --input flag):
  1. Loads data/knowledge_base.jsonl if it exists (real Wikipedia content).
  2. Falls back to a small built-in demo index otherwise.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --input data/knowledge_base.jsonl
    python scripts/build_index.py --input data/documents.jsonl data/more.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config  # noqa: E402
from modules.rag_module import RAGModule  # noqa: E402

# Default knowledge base path (populated by build_knowledge_base.py)
DEFAULT_KB_PATH = PROJECT_ROOT / "data" / "knowledge_base.jsonl"

# Minimal demo documents used when no knowledge base exists
_DEMO_DOCUMENTS = [
    {
        "text": (
            "Paris is the capital and most populous city of France. "
            "It is situated on the river Seine, in northern France, "
            "at the heart of the Ile-de-France region."
        ),
        "source": "demo_wiki_paris",
    },
    {
        "text": (
            "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France. It was named after the "
            "engineer Gustave Eiffel, whose company designed and built the tower."
        ),
        "source": "demo_wiki_eiffel",
    },
    {
        "text": (
            "Berlin is the capital and largest city of Germany by both "
            "area and population. It is situated on the river Spree in "
            "northeastern Germany."
        ),
        "source": "demo_wiki_berlin",
    },
    {
        "text": (
            "Tokyo, officially the Tokyo Metropolis, is the capital and "
            "most populous city of Japan. Located at the head of Tokyo Bay, "
            "it is the seat of the Japanese government and the Imperial Palace."
        ),
        "source": "demo_wiki_tokyo",
    },
    {
        "text": (
            "Alexander Graham Bell was a Scottish-born inventor and scientist "
            "who is credited with patenting the first practical telephone. "
            "He received the patent for the telephone on March 7, 1876."
        ),
        "source": "demo_wiki_bell",
    },
    {
        "text": (
            "Thomas Edison was an American inventor and businessman who developed "
            "many devices in fields such as electric power generation, mass "
            "communication, sound recording, and motion pictures. "
            "He is widely credited with developing the first practical incandescent light bulb."
        ),
        "source": "demo_wiki_edison",
    },
    {
        "text": (
            "The 2008 financial crisis was the worst global financial crisis "
            "since the Great Depression of the 1930s. It was triggered by a "
            "liquidity shortfall in the US banking system and resulted from "
            "the collapse of the US subprime mortgage market."
        ),
        "source": "demo_wiki_2008crisis",
    },
    {
        "text": (
            "Photosynthesis is the process used by plants, algae, and certain "
            "bacteria to harness energy from sunlight and turn it into chemical "
            "energy. It converts carbon dioxide and water into glucose and oxygen."
        ),
        "source": "demo_wiki_photosynthesis",
    },
    {
        "text": (
            "Climate change refers to long-term shifts in temperatures and "
            "weather patterns, mainly caused by human activities, especially "
            "the burning of fossil fuels. The greenhouse effect traps heat "
            "in Earth's atmosphere."
        ),
        "source": "demo_wiki_climate",
    },
    {
        "text": (
            "Albert Einstein was a German-born theoretical physicist who "
            "developed the theory of relativity, one of the two pillars of "
            "modern physics. His mass-energy equivalence formula E = mc^2 "
            "has been called the world's most famous equation."
        ),
        "source": "demo_wiki_einstein",
    },
]


def load_documents(path: str) -> list[dict]:
    """Load documents from a JSONL file.

    Each line must be a JSON object with at least a ``text`` field
    and optionally a ``source`` field.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of document dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    documents: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if "text" not in doc:
                    print(f"  WARNING: line {line_num} has no 'text' field, skipping.")
                    continue
                if "source" not in doc:
                    doc["source"] = f"line_{line_num}"
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {line_num} is invalid JSON: {e}")

    return documents


def main() -> None:
    """Entry point for the build_index script."""
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a JSONL document file."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Path(s) to JSONL file(s) with documents. "
            "If omitted, loads data/knowledge_base.jsonl when it exists, "
            "otherwise builds a small demo index."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rag = RAGModule(config=config)

    if args.input:
        # Explicit paths provided
        documents: list[dict] = []
        for inp in args.input:
            print(f"Loading documents from: {inp}")
            docs = load_documents(inp)
            print(f"  Loaded {len(docs)} documents.")
            documents.extend(docs)

    elif DEFAULT_KB_PATH.exists():
        # Auto-load the Wikipedia knowledge base
        print(f"Loading knowledge base from: {DEFAULT_KB_PATH}")
        documents = load_documents(str(DEFAULT_KB_PATH))
        print(f"  Loaded {len(documents)} documents from knowledge base.")

    else:
        # No knowledge base — use built-in demo documents
        print(
            "No knowledge base found at data/knowledge_base.jsonl.\n"
            "Tip: run 'python scripts/build_knowledge_base.py' to fetch real Wikipedia content.\n"
            "Using built-in demo documents for now."
        )
        documents = _DEMO_DOCUMENTS
        print(f"  Using {len(documents)} demo documents.")

    print(f"\nTotal documents to index: {len(documents)}")

    if not documents:
        print("ERROR: No documents to index.")
        sys.exit(1)

    print("Building FAISS index...")
    rag.build_index(documents)

    index_path = config.get("rag", {}).get("index_path", "data/faiss_index")
    print(f"Index saved to: {index_path}")

    # Quick smoke test — retrieve for a sample query
    print("\nSmoke test — retrieving for 'Who invented the telephone?'")
    results = rag.retrieve("Who invented the telephone?", top_k=2)
    if results:
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.4f}  source={r['source']}")
            print(f"       {r['text'][:120]}...")
    else:
        print("  (no results — index may be empty)")

    print("\nDone.")


if __name__ == "__main__":
    main()
