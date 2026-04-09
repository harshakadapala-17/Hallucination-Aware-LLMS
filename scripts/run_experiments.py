"""
Run Experiments Script
======================

Runs the full pipeline on a set of queries, collects traces, and reports
aggregate metrics (strategy distribution, average risk, verification stats).

Usage:
    python scripts/run_experiments.py [--queries queries.txt] [--output results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


_DEFAULT_QUERIES: List[str] = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "According to Dr. Smith, what caused the 2008 financial crisis?",
    "Compare the GDP of Japan and Germany in 2020.",
    "When was the Battle of Waterloo?",
    "What is the relationship between Einstein and relativity?",
    "How does photosynthesis work?",
    "Who cited the reference in the 2019 Nature paper?",
    "What is DNA and how does it replicate?",
    "What caused World War I and how did it impact the Treaty of Versailles?",
]


def load_queries(path: str | None) -> List[str]:
    """Load queries from a text file (one per line) or use defaults.

    Args:
        path: Optional path to a text file. If None, uses built-in queries.

    Returns:
        List of query strings.
    """
    if path is None:
        return _DEFAULT_QUERIES

    filepath = Path(path)
    if not filepath.exists():
        print(f"WARNING: Query file '{path}' not found. Using default queries.")
        return _DEFAULT_QUERIES

    queries: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries or _DEFAULT_QUERIES


def print_report(traces: List[Dict[str, Any]]) -> None:
    """Print a summary report of experiment results.

    Args:
        traces: List of pipeline trace dicts.
    """
    n = len(traces)
    if n == 0:
        print("No traces to report.")
        return

    # Strategy distribution
    strategy_counts = Counter(t["strategy"] for t in traces)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT REPORT  ({n} queries)")
    print(f"{'='*60}")

    print("\n  Strategy Distribution:")
    for s, c in strategy_counts.most_common():
        print(f"    {s:<20s}  {c:>3d}  ({c/n:.0%})")

    # Risk scores
    risk_scores = [t["prediction"]["risk_score"] for t in traces]
    print("\n  Risk Score:")
    print(f"    Mean:   {sum(risk_scores)/n:.4f}")
    print(f"    Min:    {min(risk_scores):.4f}")
    print(f"    Max:    {max(risk_scores):.4f}")

    # Verification stats
    verified_traces = [t for t in traces if t.get("verification") is not None]
    if verified_traces:
        verdicts = Counter(t["verification"]["verdict"] for t in verified_traces)
        print(f"\n  Verification Verdicts ({len(verified_traces)} verified):")
        for v, c in verdicts.most_common():
            print(f"    {v:<25s}  {c:>3d}")
    else:
        print("\n  Verification: No queries went through verification.")

    # Hallucination types
    type_counts = Counter(t["prediction"]["hallucination_type"] for t in traces)
    print("\n  Predicted Hallucination Types:")
    for ht, c in type_counts.most_common():
        print(f"    {ht:<15s}  {c:>3d}")

    print(f"\n{'='*60}")


def main() -> None:
    """Entry point for the run_experiments script."""
    parser = argparse.ArgumentParser(
        description="Run the hallucination-aware pipeline on a set of queries."
    )
    parser.add_argument(
        "--queries", type=str, default=None,
        help="Path to a text file with one query per line.",
    )
    parser.add_argument(
        "--output", type=str, default="data/experiment_results.jsonl",
        help="Path to save the full traces as JSONL.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = Pipeline(config=config)
    queries = load_queries(args.queries)

    print(f"Running {len(queries)} queries through the pipeline...")
    traces: List[Dict[str, Any]] = []

    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query[:60]}...")
        start = time.time()
        trace = pipeline.run(query)
        elapsed = time.time() - start
        trace["elapsed_seconds"] = round(elapsed, 3)
        traces.append(trace)

    # Save traces
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    print(f"\nTraces saved to: {args.output}")

    # Print report
    print_report(traces)


if __name__ == "__main__":
    main()
