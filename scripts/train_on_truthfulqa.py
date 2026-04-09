"""
Train on TruthfulQA Script
============================

Processes data/truthfulqa.jsonl into a training dataset and retrains the
HallucinationPredictor on real-labelled examples.

TruthfulQA record fields used:
  - query:            the question string
  - label:            1 = truthful (low hallucination), 0 = false claim
  - category:         topic category (e.g. "Misconceptions", "History")
  - correct_answer:   the correct answer text
  - incorrect_answers: list of wrong answers

Label mapping:
  TruthfulQA label=0 (model tends to hallucinate here) → is_hallucination=1
  TruthfulQA label=1 (model answers correctly)          → is_hallucination=0

Hallucination type is inferred from category keywords or query features.

Usage:
    python scripts/train_on_truthfulqa.py
    python scripts/train_on_truthfulqa.py --input data/truthfulqa.jsonl
    python scripts/train_on_truthfulqa.py --no-retrain   # only build dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config  # noqa: E402
from modules.hallucination_predictor import (  # noqa: E402
    HallucinationPredictor,
    _CATEGORY_TO_TYPE,
    _HIGH_RISK_CATEGORIES,
)
from modules.query_analyzer import QueryAnalyzer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_hallucination_type(
    category: str,
    features: dict,
) -> str:
    """Map a TruthfulQA category string to a hallucination type label.

    Falls back to feature-based heuristics when the category doesn't match.

    Args:
        category: TruthfulQA category string (e.g. "Misconceptions").
        features: QueryAnalyzer feature dict for the question.

    Returns:
        One of: "none", "citation", "entity", "relation", "temporal", "reasoning".
    """
    cat_lower = category.lower()
    for keyword, h_type in _CATEGORY_TO_TYPE.items():
        if keyword in cat_lower:
            return h_type

    # Feature-based fallback (matches _predict_heuristic logic)
    if features.get("contains_citation_pattern"):
        return "citation"
    if features.get("contains_date"):
        return "temporal"
    if features.get("multi_hop_indicator"):
        return "reasoning"
    if features.get("entity_count", 0) >= 2:
        return "entity"
    return "none"


def is_high_risk_category(category: str) -> bool:
    """Return True if the category is associated with high hallucination risk.

    Args:
        category: TruthfulQA category string.

    Returns:
        True if the category is in the high-risk set.
    """
    cat_lower = category.lower()
    return any(kw in cat_lower for kw in _HIGH_RISK_CATEGORIES)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training dataset from TruthfulQA and retrain predictor."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/truthfulqa.jsonl",
        help="Path to TruthfulQA JSONL file (default: data/truthfulqa.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/truthfulqa_training.jsonl",
        help="Output training JSONL path (default: data/truthfulqa_training.jsonl).",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Only build the training dataset, skip model training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    analyzer = QueryAnalyzer(config=config)

    print(f"Loading TruthfulQA from: {input_path}")
    raw_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(raw_records)} raw records.")
    print("Extracting features and building training examples ...")

    training_records = []
    errors = 0
    type_counter: Counter = Counter()
    label_counter: Counter = Counter()

    for i, record in enumerate(raw_records, 1):
        query = record.get("query", "").strip()
        if not query:
            errors += 1
            continue

        # TruthfulQA label: 1 = model answers correctly (low hallucination risk)
        #                    0 = model gives false/hallucinated answer (high risk)
        # We want is_hallucination=1 when the model tends to hallucinate.
        tqa_label = int(record.get("label", 1))
        is_hallucination = 1 - tqa_label  # invert: 0→1 (risky), 1→0 (safe)

        category = record.get("category", "")

        # Override: if category is explicitly high-risk, always mark as hallucination
        if is_high_risk_category(category):
            is_hallucination = 1

        try:
            features = analyzer.analyze(query)
        except Exception:
            errors += 1
            continue

        # Complexity proxy override when category unavailable
        if not category and features.get("complexity_score", 0) > 0.5:
            is_hallucination = 1

        h_type = infer_hallucination_type(category, features)

        training_records.append({
            "features": features,
            "is_hallucination": is_hallucination,
            "hallucination_type": h_type,
            "query": query,
            "category": category,
        })

        label_counter[is_hallucination] += 1
        type_counter[h_type] += 1

        if i % 100 == 0:
            print(f"  Processed {i}/{len(raw_records)} records ...", flush=True)

    print("\nFeature extraction complete.")
    print(f"  Total training examples: {len(training_records)}")
    print(f"  Errors / skipped:        {errors}")
    print(f"  is_hallucination=1:      {label_counter[1]}")
    print(f"  is_hallucination=0:      {label_counter[0]}")
    print("  Hallucination types:")
    for h_type, count in type_counter.most_common():
        print(f"    {h_type:<15}  {count:>4}")

    # Write training JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in training_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nTraining dataset saved to: {output_path}")

    if args.no_retrain:
        print("--no-retrain set — skipping model training.")
        return

    # ── Retrain predictor ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Training HallucinationPredictor ...")
    predictor = HallucinationPredictor(config=config)

    try:
        metrics = predictor.train(str(output_path))
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\nTraining complete.")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  F1 macro:  {metrics['f1_macro']:.4f}")
    print(f"  Models saved to: {predictor.model_path}, {predictor.type_model_path}")

    # Quick sanity check
    print("\nSanity check — predicting on 3 sample queries:")
    samples = [
        "What is the capital of France?",
        "According to ancient myth, eating watermelon seeds makes them grow inside you.",
        "How did the 2008 financial crisis compare to the Great Depression?",
    ]
    for sq in samples:
        feats = analyzer.analyze(sq)
        pred = predictor.predict(feats)
        print(
            f"  [{pred['risk_score']:.2f}] {pred['hallucination_type']:<10}  {sq[:60]}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
