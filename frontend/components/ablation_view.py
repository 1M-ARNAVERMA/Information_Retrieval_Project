"""Ablation study and system evaluation dashboard component."""

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_ablation_dashboard(ablation_data: Dict[str, Any]) -> None:
    """Renders interactive benchmark charts and tables for the 9 ablation architectures."""
    results = ablation_data.get("results", [])
    if not results:
        st.warning("No ablation results available. Run evaluation script or evaluate endpoint.")
        return

    df = pd.DataFrame(results)

    st.subheader("Architectural Ablation Benchmarks (FEVER Test Sample)")
    st.markdown(
        """
        Comparing incremental performance across all 9 retrieval and ranking configurations:
        Lexical baselines (TF-IDF, BM25) vs. Dense Bi-encoder vs. RRF Fusion vs. Entity Boosting vs. Cross-Encoder neural reranker.
        """
    )

    # 1. Comparative Grouped Bar Chart
    configs = df["Configuration"].tolist()
    p10 = df["Precision@10"].tolist()
    r10 = df["Recall@10"].tolist()
    map10 = df["MAP@10"].tolist()
    ndcg10 = df["nDCG@10"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision@10", x=configs, y=p10, marker_color="#60a5fa"))
    fig.add_trace(go.Bar(name="Recall@10", x=configs, y=r10, marker_color="#34d399"))
    fig.add_trace(go.Bar(name="MAP@10", x=configs, y=map10, marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="nDCG@10", x=configs, y=ndcg10, marker_color="#a855f7"))

    fig.update_layout(
        barmode="group",
        title=dict(text="Evaluation Metrics Across 9 Architectures", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11, color="#cbd5e1")),
        yaxis=dict(title="Score", range=[0.0, 1.05], gridcolor="#334155", tickfont=dict(color="#cbd5e1")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f8fafc")),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=100),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2. Key Insights Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="BM25 nDCG@10",
            value=f"{ndcg10[1]:.4f}",
            delta=f"+{round((ndcg10[1] - ndcg10[0]) * 100, 2)}% vs TF-IDF",
        )
    with c2:
        st.metric(
            label="RRF Fusion nDCG@10",
            value=f"{ndcg10[4]:.4f}",
            delta=f"+{round((ndcg10[4] - ndcg10[3]) * 100, 2)}% vs Merge",
        )
    with c3:
        st.metric(
            label="Full Pipeline nDCG@10",
            value=f"{ndcg10[-1]:.4f}",
            delta=f"+{round((ndcg10[-1] - ndcg10[0]) * 100, 2)}% vs Baseline",
        )

    # 3. Interactive Data Table
    st.markdown("### Metrics Summary Table")
    st.dataframe(
        df.style.highlight_max(subset=["MAP@10", "nDCG@10"], color="#065f46"),
        use_container_width=True,
    )
