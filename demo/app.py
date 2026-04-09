"""
Demo Application
=================

Streamlit-based interactive demo for the Hallucination-Aware Adaptive LLM System.

Launch with:
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from modules import load_config
from modules.generation_module import GenerationModule
from modules.hallucination_predictor import HallucinationPredictor
from modules.query_analyzer import QueryAnalyzer
from modules.rag_module import RAGModule
from modules.strategy_selector import StrategySelector
from modules.verification_module import VerificationModule
from pipeline.pipeline import Pipeline


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


# ----------------------------------------------------------------------- #
#  Main app                                                                #
# ----------------------------------------------------------------------- #

def main() -> None:
    config = _get_config()
    demo_cfg = config.get("demo", {})

    st.set_page_config(
        page_title=demo_cfg.get("title", "Hallucination-Aware LLM System"),
        page_icon="shield",
        layout="wide",
    )

    st.title(demo_cfg.get("title", "Hallucination-Aware LLM System"))
    st.markdown(
        "This system **predicts hallucination risk** before LLM generation "
        "and **routes queries** to the safest answering strategy."
    )

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        model_cfg = config.get("model", {})
        st.json({
            "Provider": model_cfg.get("provider", "openai"),
            "Model": model_cfg.get("name", "?"),
            "Low threshold": config.get("strategy", {}).get("thresholds", {}).get("low", "?"),
            "High threshold": config.get("strategy", {}).get("thresholds", {}).get("high", "?"),
            "RAG top_k": config.get("rag", {}).get("top_k", "?"),
            "HyDE": config.get("rag", {}).get("use_hyde", False),
            "NLI verification": config.get("verification", {}).get("use_nli_entailment", False),
            "Max claims": config.get("verification", {}).get("max_claims", "?"),
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

    # Main input
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

        # Run pipeline but handle streaming separately for generation
        with st.spinner("Analysing query and selecting strategy..."):
            # Run everything except generation
            analyzer = pipeline.query_analyzer
            predictor = pipeline.predictor
            selector = pipeline.selector
            rag = pipeline.rag
            verifier = pipeline.verifier

            features = analyzer.analyze(query.strip())
            prediction = predictor.predict(features)
            strategy = selector.select(prediction)

            retrieved_docs = []
            if strategy in ("rag", "rag_verification"):
                top_k = config.get("rag", {}).get("top_k", 3)
                retrieved_docs = rag.retrieve(query.strip(), top_k=top_k)

        st.divider()

        # Top metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Risk Score", f"{prediction['risk_score']:.2f}")
        with m2:
            st.metric("Strategy", _strategy_badge(strategy))
        with m3:
            st.metric("Hallucination Type", prediction.get("hallucination_type", "none"))

        # Answer — streamed or non-streamed
        st.subheader("Answer")
        context_texts = [doc["text"] for doc in retrieved_docs]

        if streaming:
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in generator.generate_stream(query.strip(), strategy, context=context_texts):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
            except Exception as e:
                full_answer = f"[Stream error: {e}]"
                answer_placeholder.markdown(full_answer)
            generation = {"answer": full_answer, "strategy_used": strategy, "prompt_tokens": 0}
        else:
            with st.spinner("Generating answer..."):
                generation = generator.generate(query.strip(), strategy, context=context_texts)
            st.info(generation["answer"])

        # Verification
        verification = None
        if strategy == "rag_verification" and retrieved_docs:
            with st.spinner("Verifying claims..."):
                verification = verifier.verify(generation["answer"], retrieved_docs)

                # Re-generate if refuted
                max_retries = config.get("pipeline", {}).get("max_regeneration_retries", 1)
                retries = 0
                while verification.get("verdict") == "refuted" and retries < max_retries:
                    st.warning("Claims were refuted — re-generating with stricter prompt...")
                    generation = generator.generate(
                        query.strip(), strategy, context=context_texts,
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

        # Verdict
        if verification:
            verdict = verification.get("verdict")
            st.metric("Verification Verdict", _verdict_badge(verdict))

        # Detail tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Features & Prediction", "Retrieved Documents",
            "Verification Details", "Full Trace",
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

        with tab2:
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(
                        f"Document {i+1} — {doc.get('source', 'unknown')} "
                        f"(score: {doc.get('score', 0):.4f})"
                    ):
                        st.write(doc.get("text", ""))
            else:
                st.info("No documents retrieved (direct_llm strategy).")

        with tab3:
            if verification:
                for c in verification.get("verified_claims", []):
                    method = c.get("method", "cosine")
                    st.success(f"[{method.upper()}] {c['claim']}")
                for c in verification.get("flagged_claims", []):
                    method = c.get("method", "cosine")
                    if c["status"] == "refuted":
                        st.error(f"[{method.upper()} / REFUTED] {c['claim']}")
                    else:
                        st.warning(f"[{method.upper()} / {c['status'].upper()}] {c['claim']}")
            else:
                st.info("Verification was not applied for this query.")

        with tab4:
            trace = {
                "query": query.strip(),
                "features": features,
                "prediction": prediction,
                "strategy": strategy,
                "retrieved_docs": retrieved_docs,
                "generation": generation,
                "verification": verification,
            }
            st.json(json.loads(json.dumps(trace, default=str)))

    elif run_btn:
        st.warning("Please enter a query first.")


if __name__ == "__main__":
    main()
