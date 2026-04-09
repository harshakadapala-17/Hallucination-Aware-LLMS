"""
Label Dataset Script
====================

Creates a labelled dataset for training the HallucinationPredictor.

Two modes:
  1. Synthetic (default): Generates queries from built-in samples and assigns
     labels via heuristics. Quick but imprecise.
  2. Ground-truth (--source): Loads TruthfulQA/FEVER JSONL files that have a
     real 'label' field and uses those as is_hallucination ground truth.
     This is the recommended mode for training a reliable predictor.

Usage:
    # Ground-truth mode (recommended)
    python scripts/label_dataset.py \\
        --source data/truthfulqa.jsonl data/fever.jsonl \\
        --output data/labeled_dataset.jsonl

    # Synthetic mode (fallback, no external data needed)
    python scripts/label_dataset.py --n 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config
from modules.query_analyzer import QueryAnalyzer

# ---------------------------------------------------------------------------
# Built-in sample queries for synthetic mode
# ---------------------------------------------------------------------------

_SAMPLE_QUERIES: List[str] = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "When was the Declaration of Independence signed?",
    "According to Dr. Smith, what caused the 2008 financial crisis?",
    "Compare the economies of Japan and Germany in 2020.",
    "What is the relationship between Einstein and the theory of relativity?",
    "Who founded Tesla Inc and when?",
    "What happened at the Battle of Waterloo in 1815?",
    "How does photosynthesis work?",
    "What is the speed of light?",
    "Explain quantum entanglement.",
    "Who was the first President of the United States?",
    "What is the population of Tokyo?",
    "How many moons does Jupiter have?",
    "What caused World War I?",
    "What is the GDP of India?",
    "Who discovered penicillin?",
    "When did the Berlin Wall fall?",
    "What is DNA?",
    "How does a combustion engine work?",
    "What is the tallest mountain in the world?",
    "According to recent studies, what is the effect of climate change on Arctic ice?",
    "Who cited the reference in the 2019 Nature paper on CRISPR?",
    "What is the source of the Nile River?",
    "Compare the reigns of Queen Victoria and Queen Elizabeth II.",
    "What is the relationship between supply and demand?",
    "When was the University of Oxford founded?",
    "Dr. Johnson at MIT published research on neural networks in January 2023.",
    "What caused the Great Depression and how did it impact global trade?",
    "Is the earth flat?",
]


def _generate_query_variants(base_queries: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prefixes = [
        "", "Tell me, ", "Can you explain ", "I want to know ",
        "According to sources, ", "Based on research, ",
    ]
    suffixes = [
        "", " Please be concise.", " Give details.",
        " Cite your sources.", " Compare with alternatives.",
    ]
    queries: List[str] = []
    for _ in range(n):
        base = rng.choice(base_queries)
        queries.append(f"{rng.choice(prefixes)}{base}{rng.choice(suffixes)}")
    return queries


def _heuristic_label(features: Dict[str, Any], rng: random.Random) -> Tuple[bool, str]:
    """Assign label by heuristic (synthetic mode only)."""
    complexity = features.get("complexity_score", 0.0)
    is_hall = rng.random() < (complexity * 0.8 + 0.1)
    if not is_hall:
        return False, "none"

    if features.get("contains_citation_pattern", False):
        h_type = "citation"
    elif features.get("contains_date", False):
        h_type = "temporal"
    elif features.get("multi_hop_indicator", False):
        h_type = "reasoning"
    elif features.get("entity_count", 0) >= 2:
        h_type = rng.choice(["entity", "relation"])
    else:
        h_type = rng.choice(["entity", "reasoning"])

    return True, h_type


def _ground_truth_label(
    record: Dict[str, Any], features: Dict[str, Any]
) -> Tuple[bool, str]:
    """Use the real 'label' field from TruthfulQA/FEVER records.

    Type is inferred from available metadata (verdict, incorrect_answers)
    when present, otherwise falls back to feature-based heuristic.
    """
    is_hall = bool(record.get("label", 0))

    if not is_hall:
        return False, "none"

    # FEVER records have a verdict field — map it to a type
    verdict = record.get("verdict", "")
    if verdict == "REFUTES":
        return True, "relation"
    if verdict == "NOT ENOUGH INFO":
        return True, "entity"

    # TruthfulQA: infer type from query features
    if features.get("contains_citation_pattern", False):
        return True, "citation"
    if features.get("contains_date", False):
        return True, "temporal"
    if features.get("multi_hop_indicator", False):
        return True, "reasoning"
    if (record.get("incorrect_answers") or []):
        # Lots of wrong answers → entity-level confusion
        return True, "entity"

    return True, "entity"


def _load_ground_truth_source(
    paths: List[str], n: Optional[int], seed: int, analyzer: QueryAnalyzer
) -> List[Dict[str, Any]]:
    """Load records from TruthfulQA/FEVER files and build labelled examples."""
    records: List[Dict[str, Any]] = []
    for path_str in paths:
        src = Path(path_str)
        if not src.exists():
            print(f"WARNING: Source file not found: {path_str}, skipping.")
            continue
        loaded = 0
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = rec.get("query", "")
                if not query or "label" not in rec:
                    continue
                features = analyzer.analyze(query)
                is_hall, h_type = _ground_truth_label(rec, features)
                records.append({
                    "query": query,
                    "features": features,
                    "is_hallucination": is_hall,
                    "hallucination_type": h_type,
                })
                loaded += 1
        print(f"  Loaded {loaded} labelled records from {path_str}")

    if n and len(records) > n:
        rng = random.Random(seed)
        records = rng.sample(records, n)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a labelled dataset for HallucinationPredictor training."
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--n", type=int, default=None,
        help="Max number of examples (default: all available from source, "
             "or 200 for synthetic mode).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source", type=str, nargs="+", default=None,
        help="Path(s) to JSONL files with real 'label' fields "
             "(e.g. data/truthfulqa.jsonl data/fever.jsonl). "
             "When provided, uses ground-truth labels instead of heuristics.",
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or config.get("data", {}).get(
        "labeled_dataset_path", "data/labeled_dataset.jsonl"
    )

    analyzer = QueryAnalyzer(config=config)

    if args.source:
        # --- Ground-truth mode ---
        print("Ground-truth mode: using real labels from source file(s).")
        records = _load_ground_truth_source(
            args.source, args.n, args.seed, analyzer
        )
        print(f"\nTotal records loaded: {len(records)}")
    else:
        # --- Synthetic mode ---
        n = args.n or 200
        print(f"Synthetic mode: generating {n} labelled examples from built-in queries.")
        rng = random.Random(args.seed)
        queries = _generate_query_variants(_SAMPLE_QUERIES, n, args.seed)
        records = []
        for query in queries:
            features = analyzer.analyze(query)
            is_hall, h_type = _heuristic_label(features, rng)
            records.append({
                "query": query,
                "features": features,
                "is_hallucination": is_hall,
                "hallucination_type": h_type,
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    hallucination_count = sum(1 for r in records if r["is_hallucination"])

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {output_path}")
    print(f"  Hallucination: {hallucination_count}  |  Clean: {len(records) - hallucination_count}")
    print(f"  Hallucination rate: {hallucination_count / max(len(records), 1):.1%}")


if __name__ == "__main__":
    main()
