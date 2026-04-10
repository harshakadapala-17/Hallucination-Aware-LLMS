"""
Demo Application
=================

Streamlit-based interactive demo for the Hallucination-Aware Adaptive LLM System.

Launch with:
    streamlit run demo/app.py --server.headless true
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from modules import load_config  # noqa: E402
from modules.generation_module import GenerationModule  # noqa: E402
from modules.rag_module import RAGModule  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


# ----------------------------------------------------------------------- #
#  Cached resources                                                        #
# ----------------------------------------------------------------------- #

@st.cache_resource
def _get_pipeline() -> Pipeline:
    return Pipeline(config=load_config())


@st.cache_resource
def _get_generator() -> GenerationModule:
    return GenerationModule(config=load_config())


def _get_config() -> Dict[str, Any]:
    return load_config()


# ----------------------------------------------------------------------- #
#  UI Helpers                                                              #
# ----------------------------------------------------------------------- #

def _risk_badge(score: float) -> str:
    if score < 0.3:
        return "Low Risk"
    elif score < 0.7:
        return "Medium Risk"
    return "High Risk"


def _strategy_badge(strategy: str) -> str:
    return {
        "direct_llm": "Direct LLM",
        "rag": "RAG-Augmented",
        "rag_verification": "RAG + Verification",
    }.get(strategy, strategy)


def _verdict_badge(verdict: str | None) -> str:
    if verdict is None:
        return "Skipped"
    return {
        "supported": "Supported",
        "partially_supported": "Partially Supported",
        "refuted": "Refuted",
        "unverifiable": "Unverifiable",
    }.get(verdict, verdict)


def _retrieval_badge(retrieval_strategy: str) -> str:
    return {
        "standard": "Standard",
        "multihop": "Multi-Hop",
        "decomposed": "Decomposed",
    }.get(retrieval_strategy, retrieval_strategy)


def _strategy_explanation(
    strategy: str,
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    hard_override: bool,
) -> str:
    """Return a one-sentence explanation of why this strategy was chosen."""
    if hard_override:
        if features.get("contains_citation_pattern"):
            return "Citation pattern detected → routing to RAG + Verification to check claims"
        elif features.get("multi_hop_indicator") and float(features.get("complexity_score", 0)) > 0.6:
            return "Multi-hop query detected → routing to RAG + Verification for thorough grounding"
        elif float(features.get("complexity_score", 0)) > 0.75:
            return "High complexity query → routing to RAG + Verification to reduce hallucination risk"

    if strategy == "direct_llm":
        return "Low risk query → answering directly, no retrieval needed"
    elif strategy == "rag":
        return "Medium risk query → grounding answer with retrieved evidence"
    elif strategy == "rag_verification":
        risk = prediction.get("risk_score", 0)
        return f"High risk query (score: {risk:.2f}) → retrieving evidence and verifying claims"
    return ""


# ----------------------------------------------------------------------- #
#  Retrieval orchestration (mirrors pipeline.py — needed for streaming)   #
# ----------------------------------------------------------------------- #

def _run_retrieval(
    query: str,
    strategy: str,
    features: Dict[str, Any],
    rag: RAGModule,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute advanced retrieval and return a retrieval info dict.

    Mirrors the retrieval logic in Pipeline.run() so the demo can stream
    the generation step while still using all advanced retrieval features.

    Returns dict with keys:
        retrieved_docs, retrieval_strategy, adaptive_topk_used,
        reranking_applied, compression_applied, retrieval_confidence,
        retrieval_details
    """
    retrieved_docs: List[Dict[str, Any]] = []
    retrieval_strategy = "standard"
    adaptive_topk_used = config.get("rag", {}).get("top_k", 3)
    reranking_applied = False
    compression_applied = False
    retrieval_confidence = 0.0
    retrieval_details: Dict[str, Any] = {}
    effective_strategy = strategy  # may be downgraded to direct_llm

    if strategy in ("rag", "rag_verification"):
        rag_cfg = config.get("rag", {})
        complexity = float(features.get("complexity_score", 0.5))

        # Adaptive top_k
        if rag_cfg.get("use_adaptive_topk", True):
            top_k = rag.get_adaptive_topk(complexity)
        else:
            top_k = int(rag_cfg.get("top_k", 3))
        adaptive_topk_used = top_k

        # Choose retrieval method
        multi_hop = bool(features.get("multi_hop_indicator", False))
        citation = bool(features.get("contains_citation_pattern", False))
        high_complexity = complexity > 0.7
        use_decomp = rag_cfg.get("use_query_decomposition", True)

        if use_decomp and (citation or high_complexity):
            retrieved_docs = rag.retrieve_decomposed(query, top_k=top_k)
            retrieval_strategy = "decomposed"
            retrieval_details["sub_questions"] = list(
                getattr(rag, "_last_sub_questions", [])
            )
        elif multi_hop:
            retrieved_docs = rag.retrieve_multihop(query, top_k=top_k)
            retrieval_strategy = "multihop"
            retrieval_details["follow_up_query"] = getattr(
                rag, "_last_followup_query", ""
            )
        else:
            retrieved_docs = rag.retrieve(query, top_k=top_k)
            retrieval_strategy = "standard"

        # Re-ranking
        if rag_cfg.get("use_reranking", True) and retrieved_docs:
            retrieved_docs = rag.rerank(query, retrieved_docs)
            reranking_applied = True

        # Confidence check — mirrors pipeline.py: suppress fallback when
        # hard_override is active (citation, multi-hop+complexity, or very
        # high complexity queries must stay grounded even with weak retrieval).
        retrieval_confidence = rag.get_retrieval_confidence(retrieved_docs)
        min_conf = float(rag_cfg.get("min_retrieval_confidence", 0.3))

        _complexity = float(features.get("complexity_score", 0.0))
        hard_override = bool(
            (features.get("multi_hop_indicator") and _complexity > 0.6)
            or features.get("contains_citation_pattern")
            or (_complexity > 0.75)
        )

        if retrieved_docs and retrieval_confidence < min_conf and not hard_override:
            # Poor retrieval and no hard override: fall back to direct LLM.
            # Keep retrieval_strategy and retrieval_details so the UI can show
            # what was attempted before the confidence fallback.
            effective_strategy = "direct_llm"
            retrieved_docs = []
            retrieval_details["confidence_fallback"] = True
        elif retrieved_docs and retrieval_confidence < min_conf and hard_override:
            # Low confidence but hard override active — keep docs, flag it.
            retrieval_details["low_confidence_override"] = True

        if rag_cfg.get("use_contextual_compression", True) and retrieved_docs:
            retrieved_docs = rag.compress_context(query, retrieved_docs)
            compression_applied = True

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_strategy": retrieval_strategy,
        "adaptive_topk_used": adaptive_topk_used,
        "reranking_applied": reranking_applied,
        "compression_applied": compression_applied,
        "retrieval_confidence": retrieval_confidence,
        "retrieval_details": retrieval_details,
        "effective_strategy": effective_strategy,
        "hard_override": hard_override,
    }


# ----------------------------------------------------------------------- #
#  Main app                                                                #
# ----------------------------------------------------------------------- #

def main() -> None:
    config = _get_config()
    demo_cfg = config.get("demo", {})
    rag_cfg = config.get("rag", {})

    st.set_page_config(
        page_title=demo_cfg.get("title", "Hallucination-Aware LLM System"),
        page_icon="shield",
        layout="wide",
    )

    st.title(demo_cfg.get("title", "Hallucination-Aware LLM System"))
    st.markdown(
        "This system **predicts hallucination risk** before LLM generation "
        "and **routes queries** to the safest answering strategy with "
        "**advanced retrieval intelligence**."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        model_cfg = config.get("model", {})
        st.json({
            "Provider": model_cfg.get("provider", "ollama"),
            "Model": model_cfg.get("name", "?"),
            "Low threshold": config.get("strategy", {}).get("thresholds", {}).get("low", "?"),
            "High threshold": config.get("strategy", {}).get("thresholds", {}).get("high", "?"),
            "NLI verification": config.get("verification", {}).get("use_nli_entailment", False),
        })

        st.divider()
        st.subheader("RAG Settings")
        st.json({
            "Adaptive top_k": rag_cfg.get("use_adaptive_topk", True),
            "top_k (low/med/high)": (
                f"{rag_cfg.get('adaptive_topk', {}).get('low', 2)} / "
                f"{rag_cfg.get('adaptive_topk', {}).get('medium', 3)} / "
                f"{rag_cfg.get('adaptive_topk', {}).get('high', 5)}"
            ),
            "Re-ranking": rag_cfg.get("use_reranking", True),
            "Query decomposition": rag_cfg.get("use_query_decomposition", True),
            "Compression": rag_cfg.get("use_contextual_compression", True),
            "Compression threshold": rag_cfg.get("compression_threshold", 0.4),
            "HyDE": rag_cfg.get("use_hyde", False),
            "Min confidence": rag_cfg.get("min_retrieval_confidence", 0.3),
        })

        st.divider()
        streaming = st.toggle("Stream answer (word by word)", value=True)

        st.divider()
        st.markdown("### Sample Queries")
        samples = [
            "What is the capital of France?",
            "According to Dr. Smith, what caused the 2008 crisis?",
            "Compare the GDPs of Japan and Germany in 2020.",
            "How does photosynthesis work?",
            "Who invented the telephone?",
            "What are the main causes of climate change?",
        ]
        for sample in samples:
            if st.button(sample, key=f"sample_{hash(sample)}"):
                st.session_state["query_input"] = sample

    # ── Main input ───────────────────────────────────────────────────────
    query = st.text_area(
        "Enter your query:",
        value=st.session_state.get("query_input", ""),
        height=100,
        placeholder="Type a question here...",
        key="query_box",
    )

    run_btn = st.button("Run Pipeline", type="primary")

    if run_btn and query.strip():
        pipeline = _get_pipeline()
        generator = _get_generator()

        with st.spinner("Analysing query and selecting strategy..."):
            analyzer = pipeline.query_analyzer
            predictor = pipeline.predictor
            selector = pipeline.selector
            rag = pipeline.rag
            verifier = pipeline.verifier

            features = analyzer.analyze(query.strip())
            # Pass query so self-consistency fires when enabled in config
            prediction = predictor.predict(features, query=query.strip())
            # Pass features to enable hard overrides for complex queries
            strategy = selector.select(prediction, features=features)
            # Mirror pipeline.py hard_override logic for badge + explanation
            _complexity = float(features.get("complexity_score", 0.0))
            hard_override = bool(
                (features.get("multi_hop_indicator") and _complexity > 0.6)
                or features.get("contains_citation_pattern")
                or (_complexity > 0.75)
            )

        with st.spinner("Retrieving documents with advanced retrieval..."):
            retrieval_info = _run_retrieval(
                query.strip(), strategy, features, rag, config
            )

        retrieved_docs = retrieval_info["retrieved_docs"]
        retrieval_strategy = retrieval_info["retrieval_strategy"]
        adaptive_topk_used = retrieval_info["adaptive_topk_used"]
        reranking_applied = retrieval_info["reranking_applied"]
        compression_applied = retrieval_info["compression_applied"]
        retrieval_confidence = retrieval_info["retrieval_confidence"]
        retrieval_details = retrieval_info["retrieval_details"]
        effective_strategy = retrieval_info["effective_strategy"]
        # Use hard_override from _run_retrieval — it is authoritative since it
        # governed the actual fallback decision inside retrieval.
        hard_override = retrieval_info.get("hard_override", hard_override)

        st.divider()

        # ── Top metrics (4 columns) ───────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Risk Score", f"{prediction['risk_score']:.2f}")
        with m2:
            st.metric("Strategy", _strategy_badge(effective_strategy))
            # Improvement 3: hard override badge — green callout shows system
            # was smart enough to override the risk-score routing.
            if hard_override:
                _feat = features  # alias for brevity
                if _feat.get("contains_citation_pattern"):
                    st.success("⚡ Override: Citation pattern")
                elif _feat.get("multi_hop_indicator"):
                    st.success("⚡ Override: Multi-hop detected")
                elif float(_feat.get("complexity_score", 0)) > 0.75:
                    st.success("⚡ Override: High complexity")
        with m3:
            st.metric("Hallucination Type", prediction.get("hallucination_type", "none"))
        with m4:
            conf_display = f"{retrieval_confidence:.2f}" if retrieval_confidence > 0 else "N/A"
            st.metric("Retrieval Confidence", conf_display)

        # Improvement 1: strategy explanation caption below the metrics row.
        _explanation = _strategy_explanation(effective_strategy, features, prediction, hard_override)
        if _explanation:
            st.caption(f"<div style='text-align:center'>{_explanation}</div>", unsafe_allow_html=True)

        # ── Answer ───────────────────────────────────────────────────────
        st.subheader("Answer")
        context_texts = [doc["text"] for doc in retrieved_docs]

        if streaming:
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in generator.generate_stream(
                    query.strip(), effective_strategy, context=context_texts
                ):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "\u258c")
                answer_placeholder.markdown(full_answer)
            except Exception as e:
                full_answer = f"[Stream error: {e}]"
                answer_placeholder.markdown(full_answer)
            generation: Dict[str, Any] = {
                "answer": full_answer,
                "strategy_used": effective_strategy,
                "prompt_tokens": 0,
            }
        else:
            with st.spinner("Generating answer..."):
                generation = generator.generate(
                    query.strip(), effective_strategy, context=context_texts
                )
            st.info(generation["answer"])

        # ── Verification ─────────────────────────────────────────────────
        verification: Optional[Dict[str, Any]] = None
        if effective_strategy == "rag_verification" and retrieved_docs:
            with st.spinner("Verifying claims..."):
                verification = verifier.verify(generation["answer"], retrieved_docs)

                max_retries = config.get("pipeline", {}).get("max_regeneration_retries", 1)
                retries = 0
                while verification.get("verdict") == "refuted" and retries < max_retries:
                    st.warning("Claims were refuted — re-generating with stricter prompt...")
                    generation = generator.generate(
                        query.strip(), effective_strategy, context=context_texts,
                        retry_hint=(
                            "Your previous answer contained claims that could not be verified. "
                            "Answer ONLY using the provided documents."
                        ),
                    )
                    verification = verifier.verify(generation["answer"], retrieved_docs)
                    retries += 1
                    if retries > 0 and streaming:
                        st.subheader("Revised Answer")
                        st.info(generation["answer"])

        if verification:
            verdict = verification.get("verdict")
            st.metric("Verification Verdict", _verdict_badge(verdict))
        else:
            verdict = None

        # Improvement 2: mini baseline vs pipeline comparison panel.
        # Collapsed by default so it doesn't clutter the main view.
        # Computes uncertainty_risk_score using the same scale as compare_baseline.py.
        verified = verification is not None
        if verdict == "supported":
            pipeline_risk = 0.0
        elif verdict == "partially_supported":
            pipeline_risk = 0.3
        elif verdict == "refuted":
            pipeline_risk = 0.8
        elif effective_strategy != "direct_llm" and not verified:
            pipeline_risk = 0.4  # RAG-grounded, not formally verified
        else:
            pipeline_risk = 0.7  # direct_llm, completely unverified

        with st.expander("📊 Baseline vs Pipeline Comparison", expanded=False):
            bc_left, bc_right = st.columns(2)
            with bc_left:
                st.markdown("**Without Pipeline (Baseline)**")
                st.metric("Uncertainty Risk", "1.00")
                st.write("No grounding applied")
                st.write("No verification applied")
                st.write("Answer returned with zero fact-checking")
            with bc_right:
                st.markdown("**With This Pipeline**")
                delta_val = 1.0 - pipeline_risk
                st.metric(
                    "Uncertainty Risk",
                    f"{pipeline_risk:.2f}",
                    delta=f"{delta_val:.2f} reduction",
                    delta_color="inverse",
                )
                rag_label = "✅ Applied" if effective_strategy != "direct_llm" else "➖ Not needed"
                st.write(f"RAG grounding: {rag_label}")
                verif_label = "✅ Applied" if verified else "➖ Not triggered"
                st.write(f"Verification: {verif_label}")
                if verdict is not None:
                    st.write(f"Verdict: {_verdict_badge(verdict)}")
            st.caption(
                "Baseline risk is always 1.0 — unverified LLM answers have maximum uncertainty. "
                "This pipeline reduces risk through grounding and verification."
            )

        # ── Detail tabs ──────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Features & Prediction",
            "Retrieved Documents",
            "Retrieval Intelligence",
            "Verification Details",
            "Full Trace",
        ])

        with tab1:
            st.subheader("Query Features")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.metric("Entity Count", features["entity_count"])
                st.metric("Token Count", features["query_length_tokens"])
                st.metric("Complexity", f"{features['complexity_score']:.4f}")
            with fc2:
                st.metric("Contains Date", "Yes" if features["contains_date"] else "No")
                st.metric("Citation Pattern", "Yes" if features["contains_citation_pattern"] else "No")
                st.metric("Multi-hop", "Yes" if features["multi_hop_indicator"] else "No")
            st.json(features.get("entity_type_flags", {}))

            # ── Self-consistency / hybrid scoring ─────────────────────
            st.divider()
            st.subheader("Risk Scoring")
            hybrid = prediction.get("hybrid_scoring", False)
            if hybrid:
                sc = prediction.get("self_consistency", {})
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    st.metric(
                        "Self-Consistency Score",
                        f"{sc.get('consistency_score', 0):.3f}",
                        help="1.0 = all Ollama samples agreed, 0.0 = completely inconsistent",
                    )
                with sc_col2:
                    st.metric(
                        "Feature Risk Score",
                        f"{prediction.get('feature_risk_score', prediction['risk_score']):.3f}",
                        help="Risk score from feature-only model (before blending)",
                    )
                n_samp = sc.get("n_samples", 0)
                st.caption(
                    f"Risk score computed using **self-consistency + features** "
                    f"({n_samp} Ollama samples at temperatures 0.3→0.9). "
                    f"Weights: features={config.get('predictor', {}).get('feature_weight', 0.4)}, "
                    f"self-consistency={config.get('predictor', {}).get('self_consistency_weight', 0.6)}."
                )
                # Show sampled answers in an expander
                answers = sc.get("answers", [])
                if answers:
                    with st.expander(f"Sampled answers ({len(answers)})"):
                        for ai, ans in enumerate(answers, 1):
                            st.markdown(f"**Sample {ai}:** {ans[:300]}")
            else:
                thresh = config.get("predictor", {}).get("self_consistency_threshold", 0.3)
                complexity = features.get("complexity_score", 0.0)
                if complexity <= thresh:
                    st.caption(
                        f"Risk score computed using **features only** "
                        f"(query complexity {complexity:.2f} ≤ threshold {thresh:.2f}, "
                        "self-consistency skipped to save latency)."
                    )
                else:
                    st.caption(
                        "Risk score computed using **features only** "
                        "(self-consistency disabled in config)."
                    )

        with tab2:
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    rerank_score = doc.get("rerank_score")
                    score_str = f"{doc.get('score', 0):.4f}"
                    if rerank_score is not None:
                        score_str += f" | rerank: {rerank_score:.2f}"
                    compressed_flag = " [compressed]" if doc.get("compressed") else ""
                    with st.expander(
                        f"Document {i+1} — {doc.get('source', 'unknown')} "
                        f"(score: {score_str}){compressed_flag}"
                    ):
                        st.write(doc.get("text", ""))
            else:
                st.info("No documents retrieved (direct_llm strategy).")

        # ── NEW: Retrieval Intelligence tab ──────────────────────────────
        with tab3:
            st.subheader("Retrieval Intelligence")

            ri1, ri2, ri3 = st.columns(3)
            with ri1:
                st.metric("Retrieval Strategy", _retrieval_badge(retrieval_strategy))
            with ri2:
                st.metric("Adaptive top_k", adaptive_topk_used)
            with ri3:
                st.metric(
                    "Retrieval Confidence",
                    f"{retrieval_confidence:.2f}" if retrieval_confidence > 0 else "N/A",
                )

            # Confidence progress bar
            if retrieval_confidence > 0:
                st.markdown("**Confidence score:**")
                st.progress(retrieval_confidence)
                min_conf = float(rag_cfg.get("min_retrieval_confidence", 0.3))
                if retrieval_confidence < min_conf:
                    st.warning(
                        f"Confidence {retrieval_confidence:.2f} below threshold "
                        f"{min_conf:.2f} — fell back to direct LLM."
                    )
                elif retrieval_confidence < 0.5:
                    st.warning("Low retrieval confidence — answers may be less grounded.")
                else:
                    st.success("Good retrieval confidence.")

            st.divider()

            # Badges
            badge_col1, badge_col2 = st.columns(2)
            with badge_col1:
                st.markdown(
                    f"**Re-ranking applied:** {'Yes' if reranking_applied else 'No'}"
                )
                st.markdown(
                    f"**Compression applied:** {'Yes' if compression_applied else 'No'}"
                )
            with badge_col2:
                st.markdown(
                    f"**Re-ranking model:** {rag_cfg.get('reranking_model', 'N/A') if reranking_applied else 'N/A'}"
                )
                compression_thresh = rag_cfg.get("compression_threshold", 0.4)
                st.markdown(
                    f"**Compression threshold:** {compression_thresh if compression_applied else 'N/A'}"
                )

            # Confidence fallback banner — shown before sub-question/follow-up details
            if retrieval_details.get("confidence_fallback"):
                attempted = _retrieval_badge(retrieval_strategy)
                st.warning(
                    f"Retrieval confidence too low ({retrieval_confidence:.2f} < "
                    f"{float(rag_cfg.get('min_retrieval_confidence', 0.3)):.2f}) — "
                    f"fell back to Direct LLM. "
                    f"The {attempted} retrieval attempted below produced results that "
                    "were not reliable enough to use as context."
                )

            if retrieval_strategy == "decomposed" and retrieval_details.get("sub_questions"):
                st.divider()
                st.markdown("**Sub-questions generated by decomposition:**")
                for i, sub_q in enumerate(retrieval_details["sub_questions"], 1):
                    st.markdown(f"  {i}. {sub_q}")

            if retrieval_strategy == "multihop" and retrieval_details.get("follow_up_query"):
                st.divider()
                st.markdown("**Follow-up query used in hop 2:**")
                st.code(retrieval_details["follow_up_query"])

        with tab4:
            if verification:
                for c in verification.get("verified_claims", []):
                    method = c.get("method", "cosine")
                    st.success(f"[{method.upper()}] {c['claim']}")
                for c in verification.get("flagged_claims", []):
                    method = c.get("method", "cosine")
                    if c["status"] == "refuted":
                        st.error(f"[{method.upper()} / REFUTED] {c['claim']}")
                    else:
                        st.warning(
                            f"[{method.upper()} / {c['status'].upper()}] {c['claim']}"
                        )
            else:
                st.info("Verification was not applied for this query.")

        with tab5:
            trace = {
                "query": query.strip(),
                "features": features,
                "prediction": prediction,
                "strategy": effective_strategy,
                "hard_override_applied": hard_override,
                "retrieval_strategy": retrieval_strategy,
                "adaptive_topk_used": adaptive_topk_used,
                "reranking_applied": reranking_applied,
                "compression_applied": compression_applied,
                "retrieval_confidence": retrieval_confidence,
                "retrieval_details": retrieval_details,
                "retrieved_docs": retrieved_docs,
                "generation": generation,
                "verification": verification,
            }
            st.json(json.loads(json.dumps(trace, default=str)))

    elif run_btn:
        st.warning("Please enter a query first.")


if __name__ == "__main__":
    main()
