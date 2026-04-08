"""
Build Index Script
==================

Loads documents from a JSONL file and builds a FAISS index using RAGModule.

Usage:
    python scripts/build_index.py [--input data/documents.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config
from modules.rag_module import RAGModule


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
        help="Path(s) to JSONL file(s) with documents. "
             "If omitted, creates a small demo index.",
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
        documents = []
        for inp in args.input:
            print(f"Loading documents from: {inp}")
            documents.extend(load_documents(inp))
    else:
        print("No --input provided. Building demo index with sample documents.")
        documents = [
            {
                "text": "Paris is the capital and most populous city of France. "
                        "It is situated on the river Seine, in northern France.",
                "source": "wiki_paris",
            },
            {
                "text": "The Eiffel Tower is a wrought-iron lattice tower on the "
                        "Champ de Mars in Paris, France. It was named after the "
                        "engineer Gustave Eiffel.",
                "source": "wiki_eiffel",
            },
            {
                "text": "Berlin is the capital and largest city of Germany by both "
                        "area and population. It is situated on the river Spree.",
                "source": "wiki_berlin",
            },
            {
                "text": "Tokyo, officially the Tokyo Metropolis, is the capital and "
                        "most populous city of Japan.",
                "source": "wiki_tokyo",
            },
            {
                "text": "The Great Wall of China is a series of fortifications that "
                        "were built across the historical northern borders of ancient "
                        "Chinese states.",
                "source": "wiki_great_wall",
            },
        ]

    print(f"Total documents: {len(documents)}")

    if not documents:
        print("ERROR: No documents to index.")
        sys.exit(1)

    print("Building FAISS index...")
    rag.build_index(documents)
    print(f"Index saved to: {config.get('rag', {}).get('index_path', 'data/faiss_index')}")
    print("Done.")


if __name__ == "__main__":
    main()
