"""Main Streamlit application for FEVER-X fact verification dashboard."""

import sys
from pathlib import Path

# Add project root to sys.path so modules import seamlessly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from frontend.api_client import BackendClient
from frontend.components.ablation_view import render_ablation_dashboard
from frontend.components.evidence_card import render_evidence_cards
from frontend.components.graph_view import render_graph_view
from frontend.components.retrieval_view import render_retrieval_playground
from frontend.components.verdict_view import render_verdict_header

# Streamlit Page Config
st.set_page_config(
    page_title="FEVER-X | Explainable Fact Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 1.0));
    }
    
    /* Header hero styling */
    .fever-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .fever-subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    
    /* Tag Pills */
    .tech-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        color: #94a3b8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 6px;
        display: inline-block;
    }
    
    /* StTabs customized */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> BackendClient:
    """Creates cached backend client."""
    return BackendClient(base_url="http://localhost:8000", prefer_direct=False)


client = get_client()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=40,
    )
    st.markdown("### 🛡️ **FEVER-X Settings**")

    # Backend diagnostics badge
    health = client.get_health()
    status_color = "#10b981" if health.get("status") == "healthy" else "#ef4444"
    docs_indexed = health.get("num_indexed_documents", 0)
    device_info = health.get("device", "cpu")

    st.markdown(
        f"""
        <div style="background: rgba(15,23,42,0.8); border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: #f8fafc;">
                <span style="width: 8px; height: 8px; background: {status_color}; border-radius: 50%; display: inline-block;"></span>
                Backend Status: <span style="color: {status_color};">{health.get('status', 'offline').upper()}</span>
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                Documents Indexed: <b>{docs_indexed}</b> | Device: <b>{device_info}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### **Sample Test Claims**")
    preset_claims = [
        "Albert Einstein was a German physicist.",
        "Albert Einstein was born in France and never studied physics.",
        "Albert Einstein loved playing the saxophone on weekends in Berlin.",
    ]

    selected_preset = st.selectbox(
        "Select a benchmark claim:",
        options=["(Select a preset claim...)"] + preset_claims,
    )

    st.markdown("---")
    st.markdown("#### **Pipeline Controls**")

    top_k_evidence = st.slider("Evidence Sentences (top_k)", min_value=1, max_value=10, value=3)

    aggregation_strategy = st.selectbox(
        "NLI Aggregation Strategy",
        options=["weighted_prob", "max_confidence", "majority_vote"],
        index=0,
    )

    use_decomposition = st.checkbox("Claim Decomposition", value=True)
    use_query_expansion = st.checkbox("Entity & Synonym Expansion", value=True)
    use_colbert = st.checkbox("ColBERT Late-Interaction", value=False)

    st.markdown("---")
    with st.expander("ℹ️ About FEVER-X Architecture"):
        st.markdown(
            """
            **FEVER-X Multi-Stage Pipeline:**
            1. **Preprocessing & NER**: Rule & regex entity boundary extraction.
            2. **Decomposition**: Splitting multi-premise claims into atomic subclaims.
            3. **Multi-Retrieval**: TF-IDF, BM25, Sentence Transformer, ColBERT.
            4. **Reciprocal Rank Fusion**: Weighted consensus ranking.
            5. **Entity Boosting**: Multiplying evidence score by named entity overlap.
            6. **Cross-Encoder Reranker**: Deep query-evidence token interaction.
            7. **MMR Diversification**: Minimizing evidence redundancy.
            8. **NLI Verification**: DeBERTa/RoBERTa entailment/contradiction inference.
            9. **Knowledge Graph**: Multi-hop path finding & bridge discovery.
            10. **Explainability**: 5-factor scoring breakdown & audit trails.
            """
        )

# --- Main Application Header ---
st.markdown("<div class='fever-title'>FEVER-X Fact Verification</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='fever-subtitle'>Explainable Multi-Stage Fact Verification using Entity-Aware Neural Retrieval and Knowledge Graph Reasoning</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
        <span class='tech-pill'>⚡ 9-Stage Hybrid Retrieval</span>
        <span class='tech-pill'>🔀 Reciprocal Rank Fusion</span>
        <span class='tech-pill'>🎯 Entity Overlap Boosting</span>
        <span class='tech-pill'>🧠 Cross-Encoder Reranking</span>
        <span class='tech-pill'>🕸️ NetworkX Knowledge Graph</span>
        <span class='tech-pill'>⚖️ DeBERTa NLI Verification</span>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# User Claim Input Box
default_claim_text = (
    selected_preset
    if selected_preset != "(Select a preset claim...)"
    else "Albert Einstein was a German physicist."
)

claim_input = st.text_input(
    "Enter a factual claim to verify:",
    value=default_claim_text,
    placeholder="e.g., Albert Einstein was a German physicist.",
    key="main_claim_input",
)

col_verify, col_clear = st.columns([1, 5])
with col_verify:
    verify_clicked = st.button("🚀 Verify Claim", type="primary", use_container_width=True)

# Maintain verification state
if "verification_result" not in st.session_state:
    st.session_state.verification_result = None

if verify_clicked and claim_input.strip():
    with st.spinner("Analyzing claim, retrieving Wikipedia evidence, and constructing reasoning graph..."):
        try:
            res = client.verify_claim(
                claim=claim_input.strip(),
                top_k=top_k_evidence,
                include_graph=True,
                aggregation_strategy=aggregation_strategy,
                use_decomposition=use_decomposition,
                use_query_expansion=use_query_expansion,
                use_colbert=use_colbert,
            )
            st.session_state.verification_result = res
        except Exception as exc:
            st.error(f"Verification pipeline failed: {exc}")

# --- Multi-Tab Presentation ---
tab_verdict, tab_graph, tab_ablation, tab_retrieval = st.tabs([
    "🛡️ Verification & Evidence",
    "🕸️ Knowledge Graph & Paths",
    "📊 Ablation & Benchmarks",
    "🔍 Retrieval Sandbox",
])

# Tab 1: Verification & Evidence
with tab_verdict:
    if st.session_state.verification_result:
        res = st.session_state.verification_result
        render_verdict_header(res)
        st.markdown("---")
        render_evidence_cards(res)
    else:
        st.info("💡 Enter a claim above and click **Verify Claim** to execute the pipeline.")

# Tab 2: Knowledge Graph
with tab_graph:
    if st.session_state.verification_result:
        render_graph_view(st.session_state.verification_result)
    else:
        st.info("Run claim verification to view the generated multi-hop knowledge graph.")

# Tab 3: Ablation Benchmarks
with tab_ablation:
    ablation_data = client.get_ablation_results()
    render_ablation_dashboard(ablation_data)

# Tab 4: Retrieval Playground
with tab_retrieval:
    render_retrieval_playground(client)
