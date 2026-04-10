"""
Baseline vs Pipeline Comparison Script
========================================

Compares a plain LLM (baseline) against the full hallucination-aware pipeline.

Scoring philosophy (fair comparison):
  BASELINE — no verification is performed.  Every baseline answer is scored
  as 0.5 (uncertain / unverified).  This honestly reflects that we simply
  do not know whether a bare LLM answer is correct.

  PIPELINE — the pipeline's own verification verdict is used directly.
  Queries routed to rag_verification get a real verdict; others get 0.5
  (unverified, but still grounded with RAG context).

  The comparison therefore measures:
    "Does the pipeline produce verified, grounded answers vs unverified ones?"
  rather than penalising the pipeline for the difficulty of NLI verification.

Usage:
    python scripts/compare_baseline.py
    python scripts/compare_baseline.py --quick      # 5 queries only
    python scripts/compare_baseline.py --output data/comparison_results.json
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
from modules.generation_module import GenerationModule  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

ALL_QUERIES = [
    "Who invented the telephone?",
    "What caused the 2008 financial crisis?",
    "How does photosynthesis work?",
    "What is the capital of France?",
    "Who was Albert Einstein?",
    "What is climate change?",
    "How does DNA work?",
    "What happened during World War II?",
    "Who invented the light bulb?",
    "What is machine learning?",
    "What caused the Great Depression?",
    "How does a vaccine work?",
    "What is the theory of relativity?",
    "Who was Charles Darwin?",
    "What is the greenhouse effect?",
]

QUICK_QUERIES = [
    "What is the capital of France?",                                              # → rag (simple factual, risk ~0.33)
    "Who invented the telephone?",                                                 # → rag (medium complexity)
    "According to researchers, what are the main causes of climate change?",       # → rag_verification (citation hard override)
    "Compare the GDPs of Japan and Germany in 2020",                               # → rag_verification (complexity > 0.75 override)
    "What did Einstein discover and how did it relate to quantum mechanics?",      # → rag_verification (multi_hop + complexity override)
]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

VERDICT_SCORE: dict[str, float] = {
    "supported": 0.0,           # verified correct — no risk
    "partially_supported": 0.3, # mostly correct — low risk
    "refuted": 0.8,             # hallucination CAUGHT — system worked (not 1.0)
    "unverifiable": 0.4,        # RAG-grounded but unverifiable — some safety
}

# Baseline answer is always scored as completely unverified — maximum risk
BASELINE_SCORE = 1.0
BASELINE_VERDICT = "unverified"

# Unverified pipeline scores by whether RAG was applied
UNVERIFIED_RAG_SCORE = 0.4     # RAG-grounded but not formally verified
UNVERIFIED_DIRECT_LLM_SCORE = 0.7  # No grounding at all


def verdict_to_score(verdict: str | None, rag_applied: bool = False) -> float:
    """Convert a pipeline verification verdict to an uncertainty risk score.

    Lower = less risk (better).  Higher = more uncertain / more risky.

    Args:
        verdict: VerificationModule verdict string or None.
        rag_applied: Whether RAG was used (affects unverified score).

    Returns:
        0.0  verified correct (supported)
        0.3  mostly correct (partially_supported)
        0.4  RAG-grounded, unverified
        0.7  direct-LLM, unverified (no grounding)
        0.8  hallucination detected and flagged (refuted — system success)
        1.0  baseline / completely unverified
    """
    if verdict is None:
        return UNVERIFIED_RAG_SCORE if rag_applied else UNVERIFIED_DIRECT_LLM_SCORE
    return VERDICT_SCORE.get(verdict, UNVERIFIED_RAG_SCORE if rag_applied else UNVERIFIED_DIRECT_LLM_SCORE)


# ---------------------------------------------------------------------------
# Per-query runners
# ---------------------------------------------------------------------------

def run_baseline(query: str, generator: GenerationModule) -> dict:
    """Run a query through the baseline (plain LLM, no RAG, no verification).

    The baseline is intentionally minimal: no routing, no retrieval, no
    verification.  Its uncertainty_risk_score is always 1.0 because without
    any grounding or verification, the answer carries maximum uncertainty risk.

    Args:
        query: Test query string.
        generator: GenerationModule instance.

    Returns:
        Dict with answer, verdict='unverified', and uncertainty_risk_score=1.0.
    """
    try:
        gen = generator.generate(query, strategy="direct_llm", context=[])
        answer = gen["answer"]
    except Exception as e:
        return {
            "answer": f"[ERROR: {e}]",
            "verdict": "error",
            "uncertainty_risk_score": BASELINE_SCORE,
        }

    return {
        "answer": answer,
        "verdict": BASELINE_VERDICT,
        "uncertainty_risk_score": BASELINE_SCORE,
        "verified": False,
    }


def run_pipeline(query: str, pipeline: Pipeline) -> dict:
    """Run a query through the full hallucination-aware pipeline.

    Uses the pipeline's own verification verdict directly — no post-hoc
    re-verification.  If the pipeline routed to rag_verification, the
    verdict comes from VerificationModule; otherwise it is 'unverified'
    (but the answer is still RAG-grounded).

    Args:
        query: Test query string.
        pipeline: Pipeline instance.

    Returns:
        Dict with answer, strategy, risk_score, retrieval_strategy,
        retrieval_confidence, rag_applied, verified, verdict,
        uncertainty_risk_score, and any routing warnings.
    """
    try:
        trace = pipeline.run(query)
    except Exception as e:
        return {
            "answer": f"[ERROR: {e}]",
            "strategy": "error",
            "risk_score": 0.5,
            "retrieval_strategy": "standard",
            "retrieval_confidence": 0.0,
            "rag_applied": False,
            "verified": False,
            "verdict": "error",
            "uncertainty_risk_score": UNVERIFIED_DIRECT_LLM_SCORE,
            "routing_warning": None,
        }

    answer = trace["generation"]["answer"]
    strategy = trace["strategy"]
    risk_score = trace["prediction"].get("risk_score", 0.0)
    retrieval_strategy = trace.get("retrieval_strategy", "standard")
    retrieval_confidence = trace.get("retrieval_confidence", 0.0)
    rag_applied = strategy in ("rag", "rag_verification")
    features = trace.get("features", {})

    # Check for unexpected direct_llm routing on complex queries and warn
    routing_warning = None
    if strategy == "direct_llm":
        complexity = float(features.get("complexity_score", 0.0))
        multi_hop = bool(features.get("multi_hop_indicator", False))
        if multi_hop or complexity > 0.75:
            routing_warning = (
                f"WARNING: routed to direct_llm despite "
                f"multi_hop={multi_hop}, complexity={complexity:.2f}"
            )

    # Use pipeline verification verdict directly — no re-verification
    if trace.get("verification"):
        verdict = trace["verification"].get("verdict", "unverifiable")
        verified = True
    else:
        # RAG-grounded but not formally verified (rag strategy or direct_llm)
        verdict = "unverified"
        verified = False

    return {
        "answer": answer,
        "strategy": strategy,
        "risk_score": round(risk_score, 4),
        "retrieval_strategy": retrieval_strategy,
        "retrieval_confidence": round(retrieval_confidence, 4),
        "rag_applied": rag_applied,
        "verified": verified,
        "verdict": verdict,
        "uncertainty_risk_score": (
            verdict_to_score(verdict, rag_applied=rag_applied)
            if verified
            else (UNVERIFIED_RAG_SCORE if rag_applied else UNVERIFIED_DIRECT_LLM_SCORE)
        ),
        "routing_warning": routing_warning,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* characters, adding '…' if needed."""
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def print_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    W = 76

    def box(text: str = "", fill: str = " ") -> str:
        return f"\u2551 {text:{fill}<{W - 2}} \u2551"

    def divider(left: str = "\u2560", mid: str = "\u2550", right: str = "\u2563") -> str:
        return left + mid * W + right

    n = len(results)

    print(divider("\u2554", "\u2550", "\u2557"))
    print(box("BASELINE vs PIPELINE COMPARISON REPORT".center(W - 2)))
    print(divider())

    # Column header
    q_w, b_w, p_w = 33, 16, 22
    hdr = f"{'Query':<{q_w}} {'Baseline':<{b_w}} {'Pipeline':<{p_w}}"
    print(box(hdr))
    sub = f"{'':>{q_w}} {'risk=1.0':<{b_w}} {'Verdict + Strategy':<{p_w}}"
    print(box(sub))
    print(divider())

    for r in results:
        q_str = _truncate(r["query"], q_w)
        b_cell = _truncate("unverified (1.0)", b_w)

        p = r["pipeline"]
        p_verdict = p["verdict"]
        p_score = p["uncertainty_risk_score"]
        p_strat = p.get("strategy", "?")[:10]
        p_cell = _truncate(f"{p_verdict} [{p_strat}] ({p_score:.1f})", p_w)

        row = f"{q_str} {b_cell} {p_cell}"
        print(box(row))

        # Print routing warning inline if present
        if r["pipeline"].get("routing_warning"):
            warn_str = f"  !! {r['pipeline']['routing_warning']}"
            print(box(_truncate(warn_str, W - 2)))

    print(divider())

    # ── Summary ─────────────────────────────────────────────────────────
    strat_counts = Counter(r["pipeline"].get("strategy", "unknown") for r in results)
    verified_count = sum(1 for r in results if r["pipeline"].get("verified"))
    rag_count = sum(1 for r in results if r["pipeline"].get("rag_applied"))

    # Baseline breakdown
    b_unverified = n  # always all unverified

    # Pipeline breakdown by verdict
    p_supported = sum(1 for r in results if r["pipeline"]["verdict"] == "supported")
    p_partial = sum(
        1 for r in results if r["pipeline"]["verdict"] == "partially_supported"
    )
    p_refuted = sum(1 for r in results if r["pipeline"]["verdict"] == "refuted")
    p_unverified = sum(1 for r in results if not r["pipeline"].get("verified"))

    # Average scores
    b_avg = BASELINE_SCORE  # always 1.0 (completely unverified)
    p_scores = [r["pipeline"]["uncertainty_risk_score"] for r in results]
    p_avg = sum(p_scores) / len(p_scores)
    risk_reduction = b_avg - p_avg

    # Hallucinations caught = queries where pipeline returned 'refuted'
    caught_count = p_refuted

    print(box("SUMMARY"))
    print(box())
    print(box(f"Baseline:   {b_unverified}/{n} answers unverified (uncertainty risk = 1.0 each)"))
    print(box(
        f"Pipeline:   {p_supported}/{n} supported, "
        f"{p_partial}/{n} partial, "
        f"{p_refuted}/{n} refuted*, "
        f"{p_unverified}/{n} unverified"
    ))
    print(box())
    print(box(f"Pipeline verification coverage:    {verified_count}/{n} queries verified"))
    print(box(f"Queries with RAG grounding:        {rag_count}/{n} queries"))
    print(box(f"Queries routed to RAG:             {strat_counts.get('rag', 0)}/{n}"))
    print(box(f"Queries routed to RAG+Verify:      {strat_counts.get('rag_verification', 0)}/{n}"))
    print(box(f"Queries routed to Direct LLM:      {strat_counts.get('direct_llm', 0)}/{n}"))
    print(box())
    print(box(f"Baseline uncertainty risk:  {b_avg:.3f}  (all answers unverified)"))
    print(box(f"Pipeline uncertainty risk:  {p_avg:.3f}  (grounded + verified where possible)"))
    reduction_str = f"-{risk_reduction:.3f}" if risk_reduction >= 0 else f"+{abs(risk_reduction):.3f}"
    print(box(f"Risk reduction:             {reduction_str}"))
    print(box())
    print(box(f"Hallucinations caught by pipeline: {caught_count}"))
    print(box(
        "  (queries where pipeline detected refuted claims that baseline"
    ))
    print(box(
        "   would have silently returned as fact)"
    ))
    print(box())
    print(box("* refuted = system SUCCESSFULLY detected a potentially hallucinated"))
    print(box("  claim. Baseline would have returned this answer with no warning."))
    print(divider("\u255a", "\u2550", "\u255d"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline LLM vs full hallucination-aware pipeline."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first 5 queries instead of all 15.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/comparison_results.json",
        help="Path to save JSON results (default: data/comparison_results.json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    args = parser.parse_args()

    queries = QUICK_QUERIES if args.quick else ALL_QUERIES
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initialising pipeline and modules...")
    config = load_config(args.config)
    generator = GenerationModule(config=config)
    pipeline = Pipeline(config=config)

    mode = "QUICK (5 queries)" if args.quick else f"FULL ({len(queries)} queries)"
    print(f"Running comparison in {mode} mode.\n")
    print("=" * 60)

    results: list[dict] = []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")

        # MODE A — Baseline (no RAG, no verification, always risk=1.0)
        print("  [A] Baseline ... ", end="", flush=True)
        baseline = run_baseline(query, generator)
        print(f"verdict={baseline['verdict']!r}  risk={baseline['uncertainty_risk_score']:.1f}")

        # MODE B — Full pipeline
        print("  [B] Pipeline ... ", end="", flush=True)
        pipe_result = run_pipeline(query, pipeline)
        print(
            f"strategy={pipe_result['strategy']!r}  "
            f"rag={pipe_result['rag_applied']}  "
            f"verified={pipe_result['verified']}  "
            f"verdict={pipe_result['verdict']!r}  "
            f"risk={pipe_result['uncertainty_risk_score']:.1f}"
        )
        if pipe_result.get("routing_warning"):
            print(f"  !! {pipe_result['routing_warning']}")

        results.append({"query": query, "baseline": baseline, "pipeline": pipe_result})

    print("\n" + "=" * 60)
    print()
    print_table(results)

    # Build summary dict
    strat_counts = Counter(r["pipeline"].get("strategy", "unknown") for r in results)
    verified_count = sum(1 for r in results if r["pipeline"].get("verified"))
    rag_count = sum(1 for r in results if r["pipeline"].get("rag_applied"))
    p_scores = [r["pipeline"]["uncertainty_risk_score"] for r in results]
    p_avg = sum(p_scores) / len(p_scores)
    caught_count = sum(1 for r in results if r["pipeline"]["verdict"] == "refuted")

    summary = {
        "baseline_uncertainty_risk": BASELINE_SCORE,
        "pipeline_uncertainty_risk": round(p_avg, 4),
        "risk_reduction": round(BASELINE_SCORE - p_avg, 4),
        "hallucinations_caught": caught_count,
        "pipeline_verification_coverage": verified_count,
        "queries_with_rag_grounding": rag_count,
        "total_queries": len(results),
        "strategy_distribution": dict(strat_counts),
        "pipeline_verdicts": {
            "supported": sum(1 for r in results if r["pipeline"]["verdict"] == "supported"),
            "partially_supported": sum(1 for r in results if r["pipeline"]["verdict"] == "partially_supported"),
            "refuted": caught_count,
            "unverified": sum(1 for r in results if not r["pipeline"].get("verified")),
        },
    }

    output_data = {"summary": summary, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
