import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import load_config
from modules.hallucination_predictor import HallucinationPredictor


def main():
    config = load_config()
    predictor = HallucinationPredictor(config)

    metrics = predictor.train('data/labeled_dataset.jsonl')
    print("Training complete!")
    print(metrics)


if __name__ == "__main__":
    main()