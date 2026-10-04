"""Interactive Knowledge & Evidence Graph visualization component using Plotly."""

import math
from typing import Any, Dict, List
import networkx as nx
import plotly.graph_objects as go
import streamlit as st


def build_plotly_network(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> go.Figure:
    """Builds a responsive, interactive 2D network diagram with Plotly."""
    G = nx.Graph()

    for n in nodes:
        node_id = n.get("id")
        label = n.get("label", "ENTITY")
        text = n.get("attributes", {}).get("text", node_id)
        verdict = n.get("attributes", {}).get("verdict")
        G.add_node(node_id, label=label, text=text, verdict=verdict)

    for e in edges:
        G.add_edge(e.get("source"), e.get("target"), relation=e.get("relation", "CONNECTED_TO"))

    # Compute 2D layout coordinates with spring layout
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_hover_text = []

    for edge in G.edges(data=True):
        u, v, data = edge
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_hover_text.append(f"{u} ➔ {data.get('relation')} ➔ {v}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#64748b"),
        hoverinfo="none",
        mode="lines",
    )

    # Node traces
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_symbol = []
    node_hover = []
    node_text = []

    for node_id, data in G.nodes(data=True):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)

        lbl = data.get("label", "ENTITY")
        txt = data.get("text", node_id)
        vrd = data.get("verdict")

        if lbl == "CLAIM":
            node_color.append("#a855f7")  # Purple
            node_size.append(26)
            node_symbol.append("diamond")
            node_hover.append(f"<b>CLAIM ROOT</b><br>{txt}")
            node_text.append("Claim")
        elif lbl == "SUBCLAIM":
            node_color.append("#06b6d4")  # Cyan
            node_size.append(18)
            node_symbol.append("square")
            node_hover.append(f"<b>SUBCLAIM</b><br>{txt}")
            node_text.append("Subclaim")
        elif lbl == "ENTITY":
            node_color.append("#3b82f6")  # Blue
            node_size.append(18)
            node_symbol.append("circle")
            node_hover.append(f"<b>ENTITY</b>: {txt}")
            node_text.append(txt[:15] if len(txt) > 15 else txt)
        elif lbl == "EVIDENCE":
            if vrd == "SUPPORTS":
                node_color.append("#10b981")  # Emerald Green
            elif vrd == "REFUTES":
                node_color.append("#ef4444")  # Red
            else:
                node_color.append("#94a3b8")  # Slate Gray
            node_size.append(22)
            node_symbol.append("circle")
            node_hover.append(f"<b>EVIDENCE ({vrd})</b><br>{txt}")
            node_text.append(node_id.replace("evidence:", ""))
        else:
            node_color.append("#64748b")
            node_size.append(14)
            node_symbol.append("circle")
            node_hover.append(txt)
            node_text.append(txt[:10])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="#f1f5f9"),
        hovertext=node_hover,
        marker=dict(
            symbol=node_symbol,
            size=node_size,
            color=node_color,
            line=dict(width=2, color="#0f172a"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Interactive Evidence & Knowledge Graph",
                font=dict(size=16, color="#f8fafc"),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=460,
        ),
    )
    return fig


def render_graph_view(result: Dict[str, Any]) -> None:
    """Renders the Knowledge Graph tab with Plotly visualization and multi-hop traces."""
    graph_data = result.get("graph", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        st.info("No knowledge graph was constructed for this claim.")
        return

    st.subheader(f"Knowledge Graph ({len(nodes)} Nodes, {len(edges)} Edges)")

    # Color legend
    st.markdown(
        """
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; font-size: 13px;">
            <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 12px; background: #a855f7; display: inline-block; transform: rotate(45deg);"></span> <b>Claim</b></span>
            <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 12px; background: #06b6d4; display: inline-block;"></span> <b>Subclaim</b></span>
            <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; display: inline-block;"></span> <b>Entity</b></span>
            <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 12px; background: #10b981; border-radius: 50%; display: inline-block;"></span> <b>Supporting Evidence</b></span>
            <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 12px; background: #ef4444; border-radius: 50%; display: inline-block;"></span> <b>Refuting Evidence</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plotly interactive network
    fig = build_plotly_network(nodes, edges)
    st.plotly_chart(fig, use_container_width=True)

    # Reasoning traces and bridge entities
    explanation = result.get("explanation", {})
    traces = explanation.get("multi_hop_traces", [])
    bridges = explanation.get("bridge_entities", [])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Multi-Hop Reasoning Paths ({len(traces)}):**")
        if traces:
            for t in traces:
                path_str = " ➔ ".join(t.get("reasoning_path", []))
                st.markdown(f"- **{t.get('evidence_id')}** ({t.get('hops')} hops):<br>`{path_str}`", unsafe_allow_html=True)
        else:
            st.caption("No intermediate multi-hop paths found; direct evidence connection.")

    with c2:
        st.markdown(f"**Bridge Entities Discovered ({len(bridges)}):**")
        if bridges:
            for b in bridges:
                st.markdown(
                    f"- 🌉 **{b.get('bridge_entity')}** ({b.get('entity_type')}): "
                    f"bridges `{b.get('connected_evidence_count')}` evidence sentences."
                )
        else:
            st.caption("No multi-evidence bridge entities found.")
