# Hallucination-Aware Adaptive LLM System

![CI](https://github.com/ksyeshwanth/Hallucination-Aware-LLMS/actions/workflows/ci.yml/badge.svg)

A fully local, modular pipeline that **predicts hallucination risk before LLM generation** and routes each query to the safest answering strategy. The system analyses query complexity, predicts the likelihood and type of hallucination, then selects between direct LLM generation, retrieval-augmented generation (RAG), or RAG with NLI-based post-generation verification. Everything runs on **Ollama — no API key required** — making it suitable for air-gapped research environments, fact-sensitive Q&A tasks, and offline experimentation.

---

## How It Works

```
QueryAnalyzer
    → HallucinationPredictor  (hybrid self-consistency scoring)
        → StrategySelector    (hard overrides + risk thresholds)
            → RAGModule       (7 advanced retrieval techniques)
                → GenerationModule  (Ollama / llama3.2)
                    → VerificationModule  (NLI cross-encoder)
```

| Strategy | When triggered |
|---|---|
| `direct_llm` | Risk score < 0.20 — low-risk query, answer directly |
| `rag` | Risk score 0.20–0.50 — ground answer with retrieved evidence |
| `rag_verification` | Risk score ≥ 0.50, or citation/multi-hop/high-complexity query (hard override) |

**Hard overrides** fire before the risk threshold and guarantee grounded, verified answers for citation-heavy, multi-hop, or highly complex queries regardless of the predictor score.

---

## Key Results

| Metric | Value |
|---|---|
| Predictor training set | 817 real TruthfulQA examples |
| Predictor AUROC | 0.6338 |
| Predictor F1 | 0.3788 |
| Baseline uncertainty risk | 1.00 (all answers unverified) |
| Pipeline uncertainty risk | ~0.44 (grounded + verified where possible) |
| Risk reduction | ~56% |

> Uncertainty risk scale: 0.0 = verified correct, 0.4 = RAG-grounded unverified, 0.7 = direct LLM unverified, 0.8 = hallucination detected (system caught it), 1.0 = baseline (no checking at all).

---

## Advanced RAG Techniques

| Technique | Description |
|---|---|
| **HyDE** | Generates a hypothetical answer via Ollama to improve retrieval query quality |
| **Multi-hop retrieval** | Two-stage retrieval: extracts entity concepts from the query, then runs a follow-up search |
| **Query decomposition** | Breaks complex queries into sub-questions, retrieves for each independently |
| **Cross-encoder reranking** | Re-scores retrieved docs with `ms-marco-MiniLM-L-6-v2` for relevance ordering |
| **Contextual compression** | Filters each document to sentences relevant to the query via cosine similarity |
| **Adaptive top-k** | Adjusts number of retrieved docs by query complexity (2 / 3 / 5) |
| **Retrieval confidence** | Computes average top-doc score; falls back to direct LLM if below threshold (unless hard override active) |

---

## Knowledge Base

- **46 Wikipedia articles** fetched via the Wikipedia REST API
- **572 chunks** (200 tokens, 50-token overlap)
- Covers science, history, economics, technology, and world events
- Indexed with FAISS + `sentence-transformers/all-MiniLM-L6-v2`

---

## Project Structure

```
Hallucination-Aware-LLMS/
├── config/
│   └── config.yaml                   # All thresholds, model names, paths
├── data/
│   ├── faiss_index/                  # Built FAISS vector index
│   ├── knowledge_base.jsonl          # 572-chunk Wikipedia knowledge base
│   ├── labeled_dataset.jsonl         # Synthetic training data (1,300 examples)
│   ├── predictor.pkl                 # Trained risk classifier
│   ├── type_classifier.pkl           # Trained hallucination type classifier
│   ├── truthfulqa.jsonl              # TruthfulQA benchmark (817 examples)
│   ├── fever.jsonl                   # FEVER benchmark (2,000 examples)
│   ├── experiment_results.jsonl      # Batch pipeline evaluation output
│   └── comparison_results.json       # Baseline vs pipeline comparison output
├── modules/
│   ├── __init__.py                   # load_config() utility
│   ├── query_analyzer.py             # Feature extraction (entities, complexity, flags)
│   ├── hallucination_predictor.py    # Risk scoring + hybrid self-consistency
│   ├── strategy_selector.py          # Routes to direct_llm / rag / rag_verification
│   ├── generation_module.py          # Prompt construction + Ollama generation
│   ├── rag_module.py                 # FAISS retrieval + 7 advanced techniques
│   └── verification_module.py        # NLI claim verification
├── pipeline/
│   └── pipeline.py                   # End-to-end orchestrator
├── scripts/
│   ├── build_knowledge_base.py       # Fetch 46 Wikipedia articles into JSONL
│   ├── build_index.py                # Build FAISS index from knowledge base
│   ├── label_dataset.py              # Generate synthetic training data
│   ├── train_predictor.py            # Train risk + type classifiers
│   ├── train_on_truthfulqa.py        # Fine-tune predictor on real TruthfulQA data
│   ├── convert_truthfulqa.py         # Convert TruthfulQA to normalized format
│   ├── convert_fever.py              # Convert FEVER to normalized format
│   ├── evaluate_benchmarks.py        # Run pipeline on benchmark datasets
│   ├── run_experiments.py            # Batch pipeline evaluation
│   └── compare_baseline.py           # Baseline vs pipeline comparison report
├── tests/
│   ├── test_query_analyzer.py
│   ├── test_hallucination_predictor.py
│   ├── test_strategy_selector.py
│   ├── test_generation_module.py
│   ├── test_rag_module.py
│   ├── test_verification_module.py
│   ├── test_pipeline.py
│   └── test_benchmark_evaluation.py
├── demo/
│   └── app.py                        # Streamlit interactive demo
├── Makefile
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- `llama3.2` model pulled:
  ```bash
  ollama pull llama3.2
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Build the knowledge base and index

```bash
# Fetch 46 Wikipedia articles (572 chunks)
python scripts/build_knowledge_base.py

# Build FAISS index from the knowledge base
python scripts/build_index.py
```

### Train the predictor on real data

```bash
# Train on 817 TruthfulQA examples (requires data/truthfulqa.jsonl)
python scripts/train_on_truthfulqa.py
```

### Launch the demo

```bash
streamlit run demo/app.py
```

Opens at `http://localhost:8501`.

---

## Baseline vs Pipeline Comparison

```bash
# Quick mode — 5 representative queries
python scripts/compare_baseline.py --quick

# Full mode — 15 queries
python scripts/compare_baseline.py

# Save results to a custom path
python scripts/compare_baseline.py --output data/my_results.json
```

The comparison measures **uncertainty risk** (0.0–1.0) for each answer:

| Outcome | Risk score |
|---|---|
| Baseline (any answer) | 1.0 — completely unverified |
| Pipeline: supported | 0.0 — verified correct |
| Pipeline: partially supported | 0.3 — mostly correct |
| Pipeline: unverified with RAG | 0.4 — grounded, not formally verified |
| Pipeline: unverified direct LLM | 0.7 — no grounding |
| Pipeline: refuted | 0.8 — hallucination caught (system worked) |

---

## Running the Full Pipeline in Python

```python
from pipeline.pipeline import Pipeline

pipe = Pipeline()
trace = pipe.run("What caused the 2008 financial crisis?")

print(trace["strategy"])                    # "rag_verification"
print(trace["hard_override_applied"])       # True/False
print(trace["retrieval_strategy"])          # "decomposed" / "multihop" / "standard"
print(trace["retrieval_confidence"])        # e.g. 0.512
print(trace["generation"]["answer"])
print(trace["verification"]["verdict"])     # "supported" / "refuted" / ...
```

### Full trace format

```python
{
    "query": str,
    "features": dict,                # entity count, complexity, flags
    "prediction": dict,              # risk_score, hallucination_type, self_consistency
    "strategy": str,                 # "direct_llm" | "rag" | "rag_verification"
    "hard_override_applied": bool,   # True if routing was overridden by feature rules
    "retrieval_strategy": str,       # "standard" | "multihop" | "decomposed"
    "adaptive_topk_used": int,       # actual top-k used for retrieval
    "reranking_applied": bool,
    "compression_applied": bool,
    "retrieval_confidence": float,   # avg similarity score of top docs
    "retrieval_details": dict,       # sub-questions, follow-up query, fallback flags
    "retrieved_docs": list,
    "generation": dict,              # answer, strategy_used, prompt_tokens
    "verification": dict | None,     # verified_claims, flagged_claims, verdict
    "hybrid_scoring": bool,          # True when self-consistency was blended
    "self_consistency_score": float | None,
    "feature_risk_score": float | None,
}
```

---

## Demo Features

The Streamlit demo (`demo/app.py`) provides:

- **4-column metrics row** — Risk Score, Strategy, Hallucination Type, Retrieval Confidence
- **Strategy explanation** — a one-sentence caption explaining why the current strategy was chosen
- **Hard override badge** — green callout when the system overrode the risk score (citation, multi-hop, or high-complexity detection)
- **Baseline vs Pipeline comparison panel** — expandable side-by-side view with uncertainty risk scores and delta
- **Self-consistency section** — shows hybrid scoring breakdown, sampled Ollama answers, and blending weights when active
- **Retrieval Intelligence tab** — retrieval strategy badge, adaptive top-k, confidence progress bar, re-ranking and compression status, sub-questions and follow-up queries
- **Verification Details tab** — per-claim supported/refuted breakdown with NLI method labels
- **Full Trace tab** — complete pipeline trace as JSON

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_query_analyzer.py -v
python -m pytest tests/test_pipeline.py -v
```

---

## License

This project is for research and educational purposes.
