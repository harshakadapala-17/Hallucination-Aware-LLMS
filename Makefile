test:
	python -m pytest tests/ -v

lint:
	ruff check modules/ pipeline/ scripts/ --ignore E501

demo:
	streamlit run demo/app.py

train:
	python scripts/label_dataset.py --n 200
	python -c "from modules import load_config; from modules.hallucination_predictor import HallucinationPredictor; p = HallucinationPredictor(load_config()); print(p.train('data/labeled_dataset.jsonl'))"

index:
	python scripts/build_index.py
