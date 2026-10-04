"""Candidate retrieval playground component."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_retrieval_playground(client: Any) -> None:
    """Renders interactive multi-retriever sandbox for testing queries."""
    st.subheader("Multi-Retriever Sandbox")
    st.markdown("Inspect candidate sentences returned by TF-IDF, BM25, and Dense Vector Search.")

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            "Search Query or Proposition:",
            value="Albert Einstein Nobel Prize in Physics",
            key="retrieval_query_input",
        )
    with c2:
        top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)

    c_ret1, c_ret2, c_ret3 = st.columns(3)
    with c_ret1:
        use_tfidf = st.checkbox("TF-IDF", value=True)
    with c_ret2:
        use_bm25 = st.checkbox("BM25", value=True)
    with c_ret3:
        use_dense = st.checkbox("Sentence Transformer", value=True)

    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        use_entity_boost = st.checkbox("Apply Entity Overlap Boost", value=True)
    with c_opt2:
        use_rrf = st.checkbox("Reciprocal Rank Fusion (RRF)", value=True)

    if st.button("Search Wikipedia Corpus", type="primary", use_container_width=True):
        active_methods = []
        if use_tfidf:
            active_methods.append("tfidf")
        if use_bm25:
            active_methods.append("bm25")
        if use_dense:
            active_methods.append("sentence_transformer")

        if not active_methods:
            st.error("Please select at least one retriever.")
            return

        with st.spinner("Retrieving candidate sentences..."):
            res = client.retrieve_candidates(
                query=query,
                top_k=top_k,
                methods=active_methods,
                use_entity_boost=use_entity_boost,
                use_rrf=use_rrf,
            )

        candidates = res.get("results", [])
        duration = res.get("execution_time_ms", 0.0)

        st.success(f"Retrieved {len(candidates)} candidate sentences in {duration} ms.")

        if candidates:
            rows = []
            for rank, c in enumerate(candidates, start=1):
                rows.append({
                    "Rank": rank,
                    "Document ID": c.get("doc_id"),
                    "Sentence #": c.get("sentence_id"),
                    "Retriever": c.get("retriever"),
                    "Score": f"{float(c.get('score', 0.0)):.4f}",
                    "Text": c.get("text"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No matching sentences found in current Wikipedia index.")
