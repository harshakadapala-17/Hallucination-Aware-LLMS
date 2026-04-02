# 🛡️ Hallucination-Aware Adaptive LLM System

A modular pipeline that **predicts hallucination risk before LLM generation** and **routes queries to the safest answering strategy**. The system analyses query complexity, predicts the likelihood and type of hallucination, and dynamically selects between direct LLM generation, retrieval-augmented generation (RAG), or RAG with post-generation verification.

The architecture is fully configurable via a single `config/config.yaml` file and designed for research reproducibility. Each module is independently testable and importable, following strict interface contracts. The pipeline produces a unified trace dict that can be consumed by the Streamlit demo, batch experiments, or downstream evaluation scripts.

---

## 📁 Project Structure

```
hallucination_aware/
├── config/
│   └── config.yaml              # All thresholds, model names, paths
├── data/
│   └── .gitkeep                  # Placeholder (index + datasets go here)
├── modules/
│   ├── __init__.py               # load_config() utility
│   ├── query_analyzer.py         # Feature extraction from queries
│   ├── hallucination_predictor.py# Risk scoring + type classification
│   ├── strategy_selector.py      # Routes to direct_llm / rag / rag_verification
│   ├── generation_module.py      # LLM prompt construction + generation
│   ├── rag_module.py             # FAISS-based retrieval
│   └── verification_module.py    # Post-generation claim verification
├── pipeline/
│   ├── __init__.py
│   └── pipeline.py               # End-to-end orchestrator
├── scripts/
│   ├── build_index.py            # Build FAISS index from documents
│   ├── label_dataset.py          # Generate synthetic training data
│   └── run_experiments.py        # Batch pipeline evaluation
├── tests/
│   ├── test_query_analyzer.py
│   ├── test_hallucination_predictor.py
│   ├── test_strategy_selector.py
│   ├── test_generation_module.py
│   ├── test_rag_module.py
│   └── test_verification_module.py
├── demo/
│   └── app.py                    # Streamlit interactive demo
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Install dependencies

```bash
cd hallucination_aware
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` to set your model name, thresholds, and file paths. The defaults work out of the box for local development.

### 3. Build the FAISS index

```bash
# Build with sample documents (no input file needed)
python scripts/build_index.py

# Or provide your own documents
python scripts/build_index.py --input data/my_documents.jsonl
```

### 4. (Optional) Generate training data and train the predictor

```bash
# Generate 200 synthetic labelled examples
python scripts/label_dataset.py --n 200

# Train the hallucination predictor (uses the labelled dataset)
python -c "
from modules.hallucination_predictor import HallucinationPredictor
p = HallucinationPredictor()
metrics = p.train('data/labeled_dataset.jsonl')
print(metrics)
"
```

### 5. Set your OpenAI API key (optional)

```bash
export OPENAI_API_KEY="sk-..."
```

> Without an API key the system returns placeholder answers but still runs the full pipeline (feature extraction, risk prediction, strategy selection, retrieval, and verification all work locally).

---

## 🧪 Running Each Module Standalone

Every module has a `__main__` block for smoke testing:

```bash
python -m modules.query_analyzer
python -m modules.hallucination_predictor
python -m modules.strategy_selector
python -m modules.generation_module
python -m modules.rag_module
python -m modules.verification_module
```

---

## 🔗 Running the Full Pipeline

```bash
python -m pipeline.pipeline
```

Or in Python:

```python
from pipeline.pipeline import Pipeline

pipe = Pipeline()
trace = pipe.run("What caused the 2008 financial crisis?")
print(trace["strategy"])       # e.g. "rag_verification"
print(trace["generation"]["answer"])
print(trace["verification"])   # claim-level verification results
```

---

## 🖥️ Launching the Demo

```bash
streamlit run demo/app.py
```

The demo runs on `http://localhost:8501` and provides:
- Interactive query input with sample queries
- Real-time risk scoring and strategy selection
- Retrieved document display
- Claim-level verification results
- Full pipeline trace in JSON

---

## 📊 Running Experiments

```bash
# Run with default queries
python scripts/run_experiments.py

# Run with a custom query file
python scripts/run_experiments.py --queries my_queries.txt --output results.jsonl
```

The experiment script outputs:
- Strategy distribution
- Risk score statistics (mean, min, max)
- Verification verdict counts
- Predicted hallucination type distribution

---

## 🧪 Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual module tests
python -m pytest tests/test_query_analyzer.py -v
python -m pytest tests/test_hallucination_predictor.py -v
python -m pytest tests/test_strategy_selector.py -v
python -m pytest tests/test_generation_module.py -v
python -m pytest tests/test_rag_module.py -v
python -m pytest tests/test_verification_module.py -v
```

---

## ⚡ Pipeline Trace Format

`Pipeline.run()` returns a single dict:

```python
{
    "query": str,
    "features": Dict,           # QueryAnalyzer output
    "prediction": Dict,         # HallucinationPredictor output
    "strategy": str,            # "direct_llm" | "rag" | "rag_verification"
    "retrieved_docs": List,     # RAGModule output (empty if direct_llm)
    "generation": Dict,         # GenerationModule output
    "verification": Dict | None # VerificationModule output or None
}
```

---

## 📝 License

This project is for research purposes.
