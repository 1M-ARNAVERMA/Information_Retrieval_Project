"""Evidence sentence card and explainable breakdown visualization component."""

from typing import Any, Dict, List
import plotly.graph_objects as go
import streamlit as st


def render_evidence_breakdown_chart(breakdown: Dict[str, Any]) -> go.Figure:
    """Renders a Plotly horizontal bar chart showing the 5 explainable scoring factors."""
    categories = [
        "Lexical Overlap",
        "Semantic Similarity",
        "Entity Match",
        "Cross-Encoder Score",
        "NLI Entailment",
    ]
    values = [
        breakdown.get("lexical_score", 0.0),
        breakdown.get("semantic_score", 0.0),
        breakdown.get("entity_score", 0.0),
        breakdown.get("reranker_score", 0.0),
        breakdown.get("nli_confidence", 0.0),
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker=dict(
                color=["#60a5fa", "#a78bfa", "#34d399", "#f472b6", "#fbbf24"],
                line=dict(color="#1e293b", width=1),
            ),
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text="Multi-Factor Relevance Breakdown", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(range=[0, 1.05], gridcolor="#334155", title="Normalized Score", tickfont=dict(color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#f8fafc")),
        margin=dict(l=10, r=20, t=35, b=25),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_evidence_cards(result: Dict[str, Any]) -> None:
    """Renders each retrieved evidence sentence with NLI breakdown and factors."""
    evidence_list = result.get("top_evidence", [])
    predictions = result.get("nli_predictions", [])
    explanation = result.get("explanation", {})
    breakdowns = explanation.get("evidence_breakdowns", [])

    if not evidence_list:
        st.info("No evidence sentences were selected by the pipeline.")
        return

    st.subheader(f"Top Evidence Sentences ({len(evidence_list)})")

    # Map predictions by (doc_id, sentence_id)
    pred_map = {(p.get("doc_id"), p.get("sentence_id")): p for p in predictions}

    for idx, ev in enumerate(evidence_list, start=1):
        doc_id = ev.get("doc_id", "Unknown")
        sent_id = ev.get("sentence_id", 0)
        text = ev.get("text", "")
        rerank_score = float(ev.get("rerank_score", 0.0))
        origin = ev.get("retriever_origin", "retrieval")

        pred = pred_map.get((doc_id, sent_id), {})
        e_p = float(pred.get("entailment_prob", 0.0))
        c_p = float(pred.get("contradiction_prob", 0.0))
        n_p = float(pred.get("neutral_prob", 0.0))
        label = pred.get("predicted_label", "NEUTRAL")

        # Color based on NLI label
        if label == "SUPPORTS":
            badge_color = "#10b981"
            badge_icon = "🟢 SUPPORTS"
        elif label == "REFUTES":
            badge_color = "#ef4444"
            badge_icon = "🔴 REFUTES"
        else:
            badge_color = "#94a3b8"
            badge_icon = "⚪ NEUTRAL"

        # Evidence Card Container
        with st.expander(f"#{idx} [{doc_id} : S{sent_id}] — {badge_icon} (Rerank: {rerank_score:.3f})", expanded=(idx == 1)):
            st.markdown(
                f"""
                <div style="font-size: 16px; font-weight: 500; color: #f8fafc; margin-bottom: 12px; line-height: 1.5;">
                    "{text}"
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**NLI Prediction Probabilities:**")
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #10b981;">Entailment (Supports)</span>
                            <span style="color: #f8fafc; font-weight: 600;">{round(e_p * 100, 1)}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #ef4444;">Contradiction (Refutes)</span>
                            <span style="color: #f8fafc; font-weight: 600;">{round(c_p * 100, 1)}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94a3b8;">Neutral (Not Enough Info)</span>
                            <span style="color: #f8fafc; font-weight: 600;">{round(n_p * 100, 1)}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Retriever Origin: `{origin}` | Document: `{doc_id}` | Line: `{sent_id}`")

            with c2:
                # Find matching breakdown
                matched_bd = None
                if idx - 1 < len(breakdowns):
                    matched_bd = breakdowns[idx - 1]
                if matched_bd:
                    fig = render_evidence_breakdown_chart(matched_bd)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Detailed factor breakdown available in explain mode.")
