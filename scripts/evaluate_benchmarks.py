import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline import Pipeline


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


def evaluate(dataset, pipeline):
    total = len(dataset)
    correct = 0
    hallucination_detected = 0
    strategy_dist = {}

    for row in dataset:
        query = row.get("query", "")
        label = row.get("label", 0)

        if not query:
            continue

        try:
            result = pipeline.run(query)

            # Risk prediction → binary
            risk_score = result.get("prediction", {}).get("risk_score", 0.0)
            pred = 1 if risk_score > 0.5 else 0

            if pred == label:
                correct += 1

            if pred == 1:
                hallucination_detected += 1

            strategy = result.get("strategy", "unknown")
            strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1

        except Exception as e:
            print(f"Error processing query: {query[:50]}... -> {e}")
            continue

    return {
        "accuracy": correct / total if total else 0,
        "hallucination_rate": hallucination_detected / total if total else 0,
        "strategy_dist": strategy_dist
    }


def main():
    print("Initializing pipeline...")
    pipeline = Pipeline()

    print("Loading datasets...")
    truthfulqa = load_jsonl("data/truthfulqa.jsonl")
    fever = load_jsonl("data/fever.jsonl")

    print("Evaluating TruthfulQA...")
    truthfulqa_results = evaluate(truthfulqa, pipeline)

    print("Evaluating FEVER...")
    fever_results = evaluate(fever, pipeline)

    results = {
        "truthfulqa": truthfulqa_results,
        "fever": fever_results
    }

    output_path = "data/benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Complete!")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()