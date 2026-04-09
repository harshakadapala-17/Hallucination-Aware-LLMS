"""
Tests for benchmark data files and evaluation output
=====================================================

Covers:
  - data/truthfulqa.jsonl schema validation
  - data/fever.jsonl schema validation
  - data/benchmark_results.json structure and key presence
  - Accuracy values are in valid [0, 1] range
  - Converter output format (sample records)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
TRUTHFULQA_PATH = DATA_DIR / "truthfulqa.jsonl"
FEVER_PATH = DATA_DIR / "fever.jsonl"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _load_jsonl(path: Path, max_records: int = 5) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= max_records:
                break
    return records


# ------------------------------------------------------------------ #
#  TruthfulQA schema                                                   #
# ------------------------------------------------------------------ #

class TestTruthfulQA:
    def test_truthfulqa_jsonl_exists(self) -> None:
        assert TRUTHFULQA_PATH.exists(), f"Missing: {TRUTHFULQA_PATH}"

    def test_truthfulqa_jsonl_is_valid(self) -> None:
        records = _load_jsonl(TRUTHFULQA_PATH)
        assert len(records) > 0, "truthfulqa.jsonl is empty"
        for rec in records:
            assert "query" in rec, f"Missing 'query' key in record: {rec}"
            assert isinstance(rec["query"], str), "'query' must be a string"
            assert "label" in rec, f"Missing 'label' key in record: {rec}"
            assert rec["label"] in (0, 1), f"'label' must be 0 or 1, got {rec['label']}"

    def test_truthfulqa_has_expected_fields(self) -> None:
        records = _load_jsonl(TRUTHFULQA_PATH)
        for rec in records:
            # Either correct_answer or incorrect_answers should be present
            has_answer = "correct_answer" in rec or "incorrect_answers" in rec
            assert has_answer, f"Record missing answer fields: {rec}"

    def test_truthfulqa_record_count(self) -> None:
        count = sum(1 for _ in TRUTHFULQA_PATH.open("r", encoding="utf-8") if _.strip())
        assert count > 100, f"Expected > 100 records, got {count}"


# ------------------------------------------------------------------ #
#  FEVER schema                                                        #
# ------------------------------------------------------------------ #

class TestFEVER:
    def test_fever_jsonl_exists(self) -> None:
        assert FEVER_PATH.exists(), f"Missing: {FEVER_PATH}"

    def test_fever_jsonl_is_valid(self) -> None:
        records = _load_jsonl(FEVER_PATH)
        assert len(records) > 0, "fever.jsonl is empty"
        valid_verdicts = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
        for rec in records:
            assert "query" in rec, f"Missing 'query' key: {rec}"
            assert isinstance(rec["query"], str), "'query' must be a string"
            assert "verdict" in rec, f"Missing 'verdict' key: {rec}"
            assert rec["verdict"] in valid_verdicts, (
                f"Invalid verdict '{rec['verdict']}', expected one of {valid_verdicts}"
            )
            assert "label" in rec, f"Missing 'label' key: {rec}"
            assert rec["label"] in (0, 1), f"'label' must be 0 or 1"

    def test_fever_has_evidence_field(self) -> None:
        records = _load_jsonl(FEVER_PATH)
        for rec in records:
            assert "evidence" in rec, f"Missing 'evidence' key: {rec}"

    def test_fever_record_count(self) -> None:
        count = sum(1 for _ in FEVER_PATH.open("r", encoding="utf-8") if _.strip())
        assert count > 100, f"Expected > 100 records, got {count}"


# ------------------------------------------------------------------ #
#  benchmark_results.json structure                                    #
# ------------------------------------------------------------------ #

class TestBenchmarkResults:
    @pytest.fixture(scope="class")
    def results(self) -> dict:
        assert BENCHMARK_RESULTS_PATH.exists(), f"Missing: {BENCHMARK_RESULTS_PATH}"
        with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_benchmark_results_has_expected_keys(self, results: dict) -> None:
        assert "truthfulqa" in results, "Missing 'truthfulqa' key"
        assert "fever" in results, "Missing 'fever' key"

    def test_truthfulqa_result_has_required_fields(self, results: dict) -> None:
        tqa = results["truthfulqa"]
        assert "accuracy" in tqa
        assert "hallucination_rate" in tqa
        assert "strategy_dist" in tqa

    def test_fever_result_has_required_fields(self, results: dict) -> None:
        fev = results["fever"]
        assert "accuracy" in fev
        assert "hallucination_rate" in fev
        assert "strategy_dist" in fev

    def test_benchmark_accuracy_in_range(self, results: dict) -> None:
        assert 0.0 <= results["truthfulqa"]["accuracy"] <= 1.0
        assert 0.0 <= results["fever"]["accuracy"] <= 1.0

    def test_hallucination_rate_in_range(self, results: dict) -> None:
        assert 0.0 <= results["truthfulqa"]["hallucination_rate"] <= 1.0
        assert 0.0 <= results["fever"]["hallucination_rate"] <= 1.0

    def test_strategy_dist_contains_valid_strategies(self, results: dict) -> None:
        valid = {"direct_llm", "rag", "rag_verification"}
        for dataset_key in ("truthfulqa", "fever"):
            dist = results[dataset_key]["strategy_dist"]
            for strategy in dist:
                assert strategy in valid, f"Invalid strategy '{strategy}' in {dataset_key}"


# ------------------------------------------------------------------ #
#  Converter output format                                             #
# ------------------------------------------------------------------ #

class TestConverterOutputFormat:
    def test_truthfulqa_converter_output(self) -> None:
        """Spot-check that the first 5 TruthfulQA records match expected schema."""
        records = _load_jsonl(TRUTHFULQA_PATH, max_records=5)
        for rec in records:
            assert isinstance(rec.get("query"), str)
            assert rec.get("label") in (0, 1)

    def test_fever_converter_output(self) -> None:
        """Spot-check that the first 5 FEVER records match expected schema."""
        records = _load_jsonl(FEVER_PATH, max_records=5)
        for rec in records:
            assert isinstance(rec.get("query"), str)
            assert rec.get("label") in (0, 1)
            assert isinstance(rec.get("evidence"), str)
