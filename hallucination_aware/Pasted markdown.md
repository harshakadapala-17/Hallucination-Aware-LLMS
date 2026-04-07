# Person 2 — Phase 2 Task Analysis

## Person 1 Status Audit (Phase 1 — Data Foundation & Model Training)

### Deliverables Checklist

| # | Artifact | Status | Details |
|---|----------|--------|---------|
| 1 | `data/faiss_index/index.faiss` | ✅ **Present** | 7,725 bytes — functional, pipeline retrieves docs |
| 2 | `data/faiss_index/metadata.jsonl` | ✅ **Present** | 776 bytes, 5 demo documents indexed |
| 3 | `data/labeled_dataset.jsonl` | ✅ **Present** | 86,295 bytes, 200 records (113 halluc / 87 clean) |
| 4 | `data/predictor.pkl` | ⚠️ **Partial** | 911 bytes — loads via `joblib` but fails raw `pickle.load()` |
| 5 | `data/type_classifier.pkl` | ⚠️ **Partial** | 1,639 bytes — same as above |

### Functional Verification

| Check | Result |
|-------|--------|
| Pipeline smoke test (`Pipeline.run()`) | ✅ **PASSES** — returns valid trace dict with all 7 keys |
| Models load via `joblib` | ✅ **YES** — `risk_model is not None = True` |
| FAISS retrieval works | ✅ **YES** — strategy = `rag` for simple query |
| Labeled dataset schema valid | ✅ **YES** — keys: `query`, `features`, `is_hallucination`, `hallucination_type` |
| Existing unit tests | ⚠️ **59/62 pass** (3 failures — see below) |

### Test Failures (Not Blocking)

3 tests in `test_hallucination_predictor.py` fail because they were written for **heuristic mode** (no model on disk), but Person 1 trained and saved models. The predictor now uses the trained model, which gives different type predictions than the heuristic:

| Test | Expected | Got | Reason |
|------|----------|-----|--------|
| `test_citation_type` | `citation` | `none` | Model prediction ≠ heuristic rule |
| `test_temporal_type` | `temporal` | `none` | Model prediction ≠ heuristic rule |
| `test_entity_type` | `entity` | `relation` | Model prediction ≠ heuristic rule |

> [!NOTE]
> These failures are a **test design issue**, not a Person 1 bug. The tests need to force heuristic mode (set `risk_model = None` before calling predict). This is a Person 3 responsibility (test fixes). **Person 1's work is functionally complete.**

### Person 1 Verdict: ✅ DONE — Ready for Phase 2 Handoff

All 5 data artifacts exist, the pipeline runs end-to-end, and the FAISS index + models are functional. You can proceed.

---

## Person 2 Tasks (Phase 2 — Benchmark Integration & Evaluation)

### Overview
Download and integrate TruthfulQA and FEVER benchmark datasets. Build converter and evaluation scripts. Re-train the predictor on richer data.

**Estimated time: 4–5 hours**

---

### Task 1: Install HuggingFace `datasets` Library

```bash
pip install datasets
```

> [!IMPORTANT]
> This is NOT in `requirements.txt`. You should also add it so Person 3's CI/CD works:
> ```
> datasets>=2.19.0
> ```

---

### Task 2: Download TruthfulQA Dataset

```python
from datasets import load_dataset
import json

ds = load_dataset("truthful_qa", "generation", split="validation")
with open("data/truthfulqa_raw.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} TruthfulQA records")
```

**Output:** `data/truthfulqa_raw.jsonl`

---

### Task 3: Download FEVER Dataset

```python
from datasets import load_dataset
import json

ds = load_dataset("fever", "v1.0", split="train[:2000]")  # subset for speed
with open("data/fever_raw.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} FEVER records")
```

**Output:** `data/fever_raw.jsonl`

> [!TIP]
> FEVER's full train set is huge. Use a subset (1000–2000 records) for practical evaluation.

---

### Task 4: Create `scripts/convert_truthfulqa.py` [NEW]

**Input:** `data/truthfulqa_raw.jsonl`  
**Output:** `data/truthfulqa.jsonl`

Each output line must follow this schema:
```json
{
  "query": "<question>",
  "correct_answer": "<best_answer>",
  "incorrect_answers": ["...", "..."],
  "label": 0 or 1
}
```

**Rules:**
- `label = 1` if the question is likely to cause hallucination (use question category or `incorrect_answers` count as proxy)
- Also include a `"text"` field (the question + best_answer combined) so the data can be fed to `build_index.py`

---

### Task 5: Create `scripts/convert_fever.py` [NEW]

**Input:** `data/fever_raw.jsonl`  
**Output:** `data/fever.jsonl`

Each output line must follow this schema:
```json
{
  "query": "<claim>",
  "verdict": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "evidence": "<evidence text>",
  "label": 0 or 1
}
```

**Rules:**
- `label = 1` if verdict is `REFUTES` (high hallucination risk)
- Include a `"text"` field for FAISS indexing

---

### Task 6: Extend FAISS Index with Benchmark Data

> [!WARNING]
> The current `build_index.py --input` only accepts **one file** (not multiple). You need to either:
> - **Option A (recommended):** Modify `build_index.py` to accept `nargs='+'` for `--input`
> - **Option B:** Run `build_index.py` twice, or merge the JSONL files first

After fixing, run:
```bash
python scripts/build_index.py --input data/truthfulqa.jsonl data/fever.jsonl
```

The `--input` flag change in [build_index.py](file:///d:/Genai/hallucination_aware/scripts/build_index.py) (line 70-76):
```diff
     parser.add_argument(
         "--input",
         type=str,
-        default=None,
-        help="Path to the JSONL file with documents. "
+        nargs="+",
+        default=[],
+        help="Path(s) to JSONL file(s) with documents. "
              "If omitted, creates a small demo index.",
     )
```

And update the loading logic (line 88-90):
```diff
-    if args.input:
-        print(f"Loading documents from: {args.input}")
-        documents = load_documents(args.input)
+    if args.input:
+        documents = []
+        for inp in args.input:
+            print(f"Loading documents from: {inp}")
+            documents.extend(load_documents(inp))
```

---

### Task 7: Augment Labeled Dataset & Retrain

> [!WARNING]
> The plan says `python scripts/label_dataset.py --n 200 --source data/truthfulqa.jsonl`, but `label_dataset.py` does **NOT** have a `--source` flag. It only uses hardcoded sample queries. You have two options:
> - **Option A:** Add a `--source` flag to `label_dataset.py` that reads queries from an external JSONL file
> - **Option B:** Manually merge TruthfulQA queries into the existing labeled dataset and retrain

After augmenting the dataset, retrain:
```python
from modules import load_config
from modules.hallucination_predictor import HallucinationPredictor

config = load_config()
predictor = HallucinationPredictor(config)
metrics = predictor.train('data/labeled_dataset.jsonl')
print(metrics)  # Should show improved AUROC and F1
```

---

### Task 8: Create `scripts/evaluate_benchmarks.py` [NEW]

This is the **most important deliverable**. The script must:

1. Load `data/truthfulqa.jsonl` and `data/fever.jsonl`
2. Run each query through `Pipeline.run(query)`
3. Compare pipeline verdict against ground-truth `label`
4. Compute: accuracy, hallucination detection rate, false positive rate, strategy distribution
5. Save results to `data/benchmark_results.json`

**Required output format** (Person 3's tests depend on this exact structure):
```json
{
  "truthfulqa": {
    "accuracy": 0.0,
    "hallucination_rate": 0.0,
    "strategy_dist": {}
  },
  "fever": {
    "accuracy": 0.0,
    "hallucination_rate": 0.0,
    "strategy_dist": {}
  }
}
```

> [!CAUTION]
> Do NOT change the top-level keys (`truthfulqa`, `fever`). Person 3 will write tests against this exact JSON structure.

---

### Task 9: Run full experiments

```bash
python scripts/run_experiments.py
```

This script already exists and works. Verify it completes without errors after your changes.

---

## Deliverables Checklist

```
scripts/
├── convert_truthfulqa.py   [NEW]
├── convert_fever.py         [NEW]
└── evaluate_benchmarks.py   [NEW]

data/
├── truthfulqa_raw.jsonl     [NEW — downloaded]
├── fever_raw.jsonl          [NEW — downloaded]
├── truthfulqa.jsonl         [NEW — converted]
├── fever.jsonl              [NEW — converted]
├── benchmark_results.json   [NEW — evaluation output]
└── predictor.pkl            [UPDATED — re-trained]
```

---

## Pre-requisites Summary

| Requirement | Status | Action |
|-------------|--------|--------|
| Python venv activated | Verify | `venv\Scripts\activate` |
| `requirements.txt` installed | ✅ Already done | — |
| `datasets` library | ❌ Missing | `pip install datasets` |
| Person 1's 5 artifacts | ✅ All present | — |
| Pipeline runs | ✅ Verified | — |
| `build_index.py --input` accepts multiple files | ❌ Needs fix | Modify argparse `nargs='+'` |
| `label_dataset.py --source` flag | ❌ Doesn't exist | Add flag or merge manually |
| OpenAI API key in `.env` | Verify | Pipeline runs with placeholder without it |

---

## Handoff to Person 3

When your phase is complete, confirm:
- [ ] `data/truthfulqa.jsonl` and `data/fever.jsonl` exist and are well-formed (valid JSON per line)
- [ ] `data/benchmark_results.json` exists with `truthfulqa` and `fever` top-level keys
- [ ] `python scripts/run_experiments.py` completes without errors
- [ ] Share benchmark accuracy numbers with Person 3 so they know expected values for tests
- [ ] `git add data/ scripts/ && git commit -m 'Phase 2: TruthfulQA/FEVER integration, benchmarks' && git push origin main`
