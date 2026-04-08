"""
Label Dataset Script
====================

Creates a labelled dataset for training the HallucinationPredictor.
Generates synthetic labelled examples by running queries through
QueryAnalyzer and assigning hallucination labels using heuristics.

Usage:
    python scripts/label_dataset.py [--output data/labeled_dataset.jsonl] [--n 200]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config
from modules.query_analyzer import QueryAnalyzer

# Sample queries for synthetic data generation
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
    """Generate *n* query strings by sampling and mutating base queries.

    Args:
        base_queries: List of seed query strings.
        n: Number of queries to generate.
        seed: Random seed.

    Returns:
        List of query strings.
    """
    rng = random.Random(seed)
    queries: List[str] = []
    prefixes = [
        "", "Tell me, ", "Can you explain ", "I want to know ",
        "According to sources, ", "Based on research, ",
    ]
    suffixes = [
        "", " Please be concise.", " Give details.",
        " Cite your sources.", " Compare with alternatives.",
    ]

    for _ in range(n):
        base = rng.choice(base_queries)
        prefix = rng.choice(prefixes)
        suffix = rng.choice(suffixes)
        queries.append(f"{prefix}{base}{suffix}")

    return queries


def _assign_label(
    features: Dict[str, Any], rng: random.Random
) -> tuple[bool, str]:
    """Assign synthetic hallucination labels based on features.

    Higher complexity → higher chance of being labelled as hallucination.
    The hallucination type is heuristically assigned based on feature flags.

    Args:
        features: Feature dict from QueryAnalyzer.
        rng: Random number generator.

    Returns:
        Tuple of (is_hallucination, hallucination_type).
    """
    complexity = features.get("complexity_score", 0.0)
    # Probability of hallucination scales with complexity
    is_hall = rng.random() < (complexity * 0.8 + 0.1)

    if not is_hall:
        return False, "none"

    # Assign type based on features
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


def main() -> None:
    """Entry point for the label_dataset script."""
    parser = argparse.ArgumentParser(
        description="Generate a labelled dataset for HallucinationPredictor training."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: from config).",
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of labelled examples to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--source", type=str, nargs="+", default=None,
        help="Path(s) to external JSONL file(s) with 'query' field to use as "
             "input queries instead of built-in samples. Records are appended "
             "to existing dataset. Use with TruthfulQA/FEVER converted data.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or config.get("data", {}).get(
        "labeled_dataset_path", "data/labeled_dataset.jsonl"
    )

    analyzer = QueryAnalyzer(config=config)
    rng = random.Random(args.seed)

    # Load queries from external source files or generate from built-in samples
    if args.source:
        queries: List[str] = []
        for source_path in args.source:
            src = Path(source_path)
            if not src.exists():
                print(f"WARNING: Source file not found: {source_path}, skipping.")
                continue
            with open(src, "r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        q = record.get("query", "")
                        if q:
                            queries.append(q)
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(queries)} queries from {source_path}")
        # Limit to --n if specified
        if args.n and len(queries) > args.n:
            rng_sample = random.Random(args.seed)
            queries = rng_sample.sample(queries, args.n)
        file_mode = "a"  # Append to existing dataset
        print(f"Using {len(queries)} queries from external source(s) (append mode)")
    else:
        queries = _generate_query_variants(_SAMPLE_QUERIES, args.n, args.seed)
        file_mode = "w"  # Overwrite when generating from scratch

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    hallucination_count = 0
    total_written = 0
    with open(output_path, file_mode, encoding="utf-8") as f:
        for i, query in enumerate(queries):
            features = analyzer.analyze(query)
            is_hall, h_type = _assign_label(features, rng)
            if is_hall:
                hallucination_count += 1

            record = {
                "query": query,
                "features": features,
                "is_hallucination": is_hall,
                "hallucination_type": h_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"{'Appended' if file_mode == 'a' else 'Generated'} {total_written} labelled examples → {output_path}")
    print(f"  Hallucination: {hallucination_count}  |  Clean: {total_written - hallucination_count}")
    print(f"  Hallucination rate: {hallucination_count / max(total_written, 1):.1%}")


if __name__ == "__main__":
    main()
