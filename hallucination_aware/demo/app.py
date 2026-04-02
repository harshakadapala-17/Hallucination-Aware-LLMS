"""
Demo Application
=================

Streamlit-based interactive demo for the Hallucination-Aware Adaptive LLM System.

Launch with:
    streamlit run demo/app.py

Or from the project root:
    python -m streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from modules import load_config
from pipeline.pipeline import Pipeline


# ----------------------------------------------------------------------- #
#  Initialisation                                                          #
# ----------------------------------------------------------------------- #

@st.cache_resource
def _get_pipeline() -> Pipeline:
    """Create and cache the Pipeline instance."""
    config = load_config()
    return Pipeline(config=config)


def _get_config() -> Dict[str, Any]:
    """Load config for display purposes."""
    return load_config()


# ----------------------------------------------------------------------- #
#  UI Helpers                                                              #
# ----------------------------------------------------------------------- #

def _risk_badge(score: float) -> str:
    """Return an emoji + label for a risk score."""
    if score < 0.3:
        return "🟢 Low Risk"
    elif score < 0.7:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


def _strategy_badge(strategy: str) -> str:
    """Return an emoji + label for a strategy."""
    mapping = {
        "direct_llm": "⚡ Direct LLM",
        "rag": "📚 RAG-Augmented",
        "rag_verification": "🔍 RAG + Verification",
    }
    return mapping.get(strategy, strategy)


def _verdict_badge(verdict: str | None) -> str:
    """Return an emoji + label for a verification verdict."""
    if verdict is None:
        return "⏭️ Skipped"
    mapping = {
        "supported": "✅ Supported",
        "partially_supported": "⚠️ Partially Supported",
        "refuted": "❌ Refuted",
        "unverifiable": "❓ Unverifiable",
    }
    return mapping.get(verdict, verdict)


# ----------------------------------------------------------------------- #
#  Main app                                                                #
# ----------------------------------------------------------------------- #

def main() -> None:
    """Streamlit app entry point."""
    config = _get_config()
    demo_cfg = config.get("demo", {})

    st.set_page_config(
        page_title=demo_cfg.get("title", "Hallucination-Aware LLM System"),
        page_icon="🛡️",
        layout="wide",
    )

    # Header
    st.title("🛡️ " + demo_cfg.get("title", "Hallucination-Aware LLM System"))
    st.markdown(
        "This system **predicts hallucination risk** before LLM generation "
        "and **routes queries** to the safest answering strategy."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.json({
            "Model": config.get("model", {}).get("name", "?"),
            "Low threshold": config.get("strategy", {}).get("thresholds", {}).get("low", "?"),
            "High threshold": config.get("strategy", {}).get("thresholds", {}).get("high", "?"),
            "RAG top_k": config.get("rag", {}).get("top_k", "?"),
            "Max claims": config.get("verification", {}).get("max_claims", "?"),
        })

        st.divider()
        st.markdown("### 📋 Sample Queries")
        samples = [
            "What is the capital of France?",
            "According to Dr. Smith, what caused the 2008 crisis?",
            "Compare the GDPs of Japan and Germany in 2020.",
            "How does photosynthesis work?",
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

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run_btn and query.strip():
        pipeline = _get_pipeline()

        with st.spinner("Running pipeline..."):
            trace = pipeline.run(query.strip())

        # Results layout
        st.divider()

        # Top-level metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Risk Score", f"{trace['prediction']['risk_score']:.2f}")
        with m2:
            st.metric("Risk Level", _risk_badge(trace['prediction']['risk_score']))
        with m3:
            st.metric("Strategy", _strategy_badge(trace['strategy']))
        with m4:
            verdict = trace['verification']['verdict'] if trace['verification'] else None
            st.metric("Verification", _verdict_badge(verdict))

        # Answer
        st.subheader("💬 Answer")
        st.info(trace["generation"]["answer"])

        # Detail tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Features", "🎯 Prediction", "📚 Retrieved Docs",
            "🔍 Verification", "📝 Full Trace",
        ])

        with tab1:
            st.subheader("Query Features")
            features = trace["features"]
            fc1, fc2 = st.columns(2)
            with fc1:
                st.metric("Entity Count", features["entity_count"])
                st.metric("Token Count", features["query_length_tokens"])
                st.metric("Avg Token Length", f"{features['avg_token_length']:.2f}")
                st.metric("Complexity", f"{features['complexity_score']:.4f}")
            with fc2:
                st.metric("Contains Date", "✅" if features["contains_date"] else "❌")
                st.metric("Citation Pattern", "✅" if features["contains_citation_pattern"] else "❌")
                st.metric("Multi-hop", "✅" if features["multi_hop_indicator"] else "❌")

            st.markdown("**Entity Type Flags:**")
            st.json(features.get("entity_type_flags", {}))

        with tab2:
            st.subheader("Hallucination Prediction")
            pred = trace["prediction"]
            st.metric("Risk Score", f"{pred['risk_score']:.4f}")
            st.metric("Hallucination Type", pred["hallucination_type"])
            st.metric("Type Confidence", f"{pred['type_confidence']:.4f}")
            st.markdown(f"**Selected Strategy:** {_strategy_badge(trace['strategy'])}")

        with tab3:
            st.subheader("Retrieved Documents")
            docs = trace.get("retrieved_docs", [])
            if docs:
                for i, doc in enumerate(docs):
                    with st.expander(f"Document {i+1} — {doc.get('source', 'unknown')} (score: {doc.get('score', 0):.4f})"):
                        st.write(doc.get("text", ""))
            else:
                st.info("No documents retrieved (direct_llm strategy).")

        with tab4:
            st.subheader("Verification Results")
            ver = trace.get("verification")
            if ver:
                st.markdown(f"**Verdict:** {_verdict_badge(ver['verdict'])}")

                if ver["verified_claims"]:
                    st.markdown("**✅ Verified Claims:**")
                    for c in ver["verified_claims"]:
                        st.success(f"{c['claim']}  (sim: {c['max_similarity']:.4f})")

                if ver["flagged_claims"]:
                    st.markdown("**⚠️ Flagged Claims:**")
                    for c in ver["flagged_claims"]:
                        if c["status"] == "refuted":
                            st.error(f"{c['claim']}  (sim: {c['max_similarity']:.4f})")
                        else:
                            st.warning(f"{c['claim']}  (sim: {c['max_similarity']:.4f})")
            else:
                st.info("Verification was not applied for this query.")

        with tab5:
            st.subheader("Full Pipeline Trace")
            st.json(json.loads(json.dumps(trace, default=str)))

    elif run_btn:
        st.warning("Please enter a query first.")


if __name__ == "__main__":
    main()
