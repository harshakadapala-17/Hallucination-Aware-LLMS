"""
Convert TruthfulQA Script
=========================

Converts the raw TruthfulQA HuggingFace export into the normalised format
expected by the pipeline and evaluation scripts.

Input:  data/truthfulqa_raw.jsonl  (HuggingFace 'truthful_qa' generation split)
Output: data/truthfulqa.jsonl

Usage:
    python scripts/convert_truthfulqa.py [--input data/truthfulqa_raw.jsonl] [--output data/truthfulqa.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config  # noqa: E402


def convert_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single raw TruthfulQA record to normalised format.

    Args:
        raw: Raw record from the HuggingFace export.

    Returns:
        Normalised record dict.
    """
    question = raw.get("question", "")
    best_answer = raw.get("best_answer", "")
    incorrect_answers = raw.get("incorrect_answers", [])
    category = raw.get("category", "")

    # label = 1 if the question is likely to cause hallucination
    # Heuristic: questions with many incorrect answers or from adversarial
    # categories are high hallucination risk
    num_incorrect = len(incorrect_answers) if isinstance(incorrect_answers, list) else 0
    q_type = raw.get("type", "").lower()

    # High-risk if adversarial, many wrong answers, or from misconception categories
    is_high_risk = (
        q_type == "adversarial"
        or num_incorrect >= 4
        or "misconception" in category.lower()
        or "fiction" in category.lower()
        or "conspiracy" in category.lower()
    )
    label = 1 if is_high_risk else 0

    # Build a text field for FAISS indexing (question + best answer)
    text = f"{question} {best_answer}".strip()

    return {
        "query": question,
        "correct_answer": best_answer,
        "incorrect_answers": incorrect_answers if isinstance(incorrect_answers, list) else [],
        "label": label,
        "text": text,
        "source": "truthfulqa",
        "category": category,
    }


def main() -> None:
    """Entry point for the convert_truthfulqa script."""
    parser = argparse.ArgumentParser(
        description="Convert raw TruthfulQA data to normalised JSONL format."
    )
    parser.add_argument(
        "--input", type=str, default="data/truthfulqa_raw.jsonl",
        help="Path to the raw TruthfulQA JSONL file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: from config or data/truthfulqa.jsonl).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or config.get("data", {}).get(
        "truthfulqa_path", "data/truthfulqa.jsonl"
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Load raw records
    raw_records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    print(f"Loaded {len(raw_records)} raw TruthfulQA records from {args.input}")

    # Convert
    converted = [convert_record(r) for r in raw_records]
    label_1_count = sum(1 for r in converted if r["label"] == 1)

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records → {output_path}")
    print(f"  High hallucination risk (label=1): {label_1_count}")
    print(f"  Low hallucination risk  (label=0): {len(converted) - label_1_count}")


if __name__ == "__main__":
    main()
