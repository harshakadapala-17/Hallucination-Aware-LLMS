"""
Convert FEVER Script
====================

Converts the raw FEVER HuggingFace export into the normalised format
expected by the pipeline and evaluation scripts.

Input:  data/fever_raw.jsonl  (HuggingFace 'copenlu/fever_gold_evidence' format)
Output: data/fever.jsonl

Usage:
    python scripts/convert_fever.py [--input data/fever_raw.jsonl] [--output data/fever.jsonl]
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

from modules import load_config


def _extract_evidence_text(evidence_raw: Any) -> str:
    """Extract a human-readable evidence string from the raw FEVER format.

    The raw evidence field is a list of lists. Each inner list contains
    elements like [article_title, sentence_id, article_title, sentence_text].
    We extract and concatenate the sentence text portions.

    Args:
        evidence_raw: Raw evidence field from FEVER dataset.

    Returns:
        Concatenated evidence sentences as a single string.
    """
    if not isinstance(evidence_raw, list):
        return ""

    sentences: List[str] = []
    for evidence_group in evidence_raw:
        if isinstance(evidence_group, list) and len(evidence_group) >= 4:
            # Format: [article_title, sentence_id, article_title, sentence_text]
            sentence_text = str(evidence_group[-1]).strip()
            if sentence_text and sentence_text.lower() != "none":
                sentences.append(sentence_text)
        elif isinstance(evidence_group, list) and len(evidence_group) >= 2:
            # Fallback: try last element
            sentence_text = str(evidence_group[-1]).strip()
            if sentence_text and sentence_text.lower() != "none":
                sentences.append(sentence_text)

    return " ".join(sentences) if sentences else ""


def _normalise_verdict(label: str) -> str:
    """Normalise a FEVER label string to the canonical format.

    Args:
        label: Raw label string from FEVER.

    Returns:
        One of 'SUPPORTS', 'REFUTES', or 'NOT ENOUGH INFO'.
    """
    label_upper = str(label).upper().strip()
    if "REFUTE" in label_upper:
        return "REFUTES"
    if "SUPPORT" in label_upper:
        return "SUPPORTS"
    return "NOT ENOUGH INFO"


def convert_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single raw FEVER record to normalised format.

    Args:
        raw: Raw record from the HuggingFace export.

    Returns:
        Normalised record dict.
    """
    claim = raw.get("claim", "")
    raw_label = raw.get("label", "NOT ENOUGH INFO")
    evidence_raw = raw.get("evidence", [])

    verdict = _normalise_verdict(raw_label)
    evidence_text = _extract_evidence_text(evidence_raw)

    # label = 1 if verdict is REFUTES (high hallucination risk)
    label = 1 if verdict == "REFUTES" else 0

    # Build a text field for FAISS indexing
    text = f"{claim} {evidence_text}".strip() if evidence_text else claim

    return {
        "query": claim,
        "verdict": verdict,
        "evidence": evidence_text,
        "label": label,
        "text": text,
        "source": "fever",
    }


def main() -> None:
    """Entry point for the convert_fever script."""
    parser = argparse.ArgumentParser(
        description="Convert raw FEVER data to normalised JSONL format."
    )
    parser.add_argument(
        "--input", type=str, default="data/fever_raw.jsonl",
        help="Path to the raw FEVER JSONL file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: from config or data/fever.jsonl).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or config.get("data", {}).get(
        "fever_path", "data/fever.jsonl"
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

    print(f"Loaded {len(raw_records)} raw FEVER records from {args.input}")

    # Convert
    converted = [convert_record(r) for r in raw_records]
    label_1_count = sum(1 for r in converted if r["label"] == 1)

    # Stats
    verdict_counts: Dict[str, int] = {}
    for r in converted:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records → {output_path}")
    print(f"  Verdict distribution: {verdict_counts}")
    print(f"  High hallucination risk (label=1 / REFUTES): {label_1_count}")
    print(f"  Low hallucination risk  (label=0):           {len(converted) - label_1_count}")


if __name__ == "__main__":
    main()
