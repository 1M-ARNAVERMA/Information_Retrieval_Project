"""Verdict banner, confidence meter, and reasoning audit view component."""

from typing import Any, Dict, List
import streamlit as st


def render_verdict_header(result: Dict[str, Any]) -> None:
    """Renders the main verdict card, confidence meter, and explanation summary."""
    verdict = result.get("verdict", "NOT ENOUGH INFO")
    confidence = float(result.get("confidence", 0.5))
    conf_pct = round(confidence * 100, 1)

    # Color palette
    if verdict == "SUPPORTS":
        bg_color = "rgba(16, 185, 129, 0.15)"
        border_color = "#10b981"
        text_color = "#10b981"
        icon = "✅"
    elif verdict == "REFUTES":
        bg_color = "rgba(239, 68, 68, 0.15)"
        border_color = "#ef4444"
        text_color = "#ef4444"
        icon = "❌"
    else:
        bg_color = "rgba(245, 158, 11, 0.15)"
        border_color = "#f59e0b"
        text_color = "#f59e0b"
        icon = "⚠️"

    # Verdict Hero Banner
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">Verification Verdict</div>
                    <div style="font-size: 32px; font-weight: 800; color: {text_color}; display: flex; align-items: center; gap: 8px;">
                        <span>{icon}</span> {verdict}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">System Confidence</div>
                    <div style="font-size: 32px; font-weight: 800; color: #f8fafc;">
                        {conf_pct}%
                    </div>
                </div>
            </div>
            <div style="margin-top: 14px; background: rgba(255,255,255,0.1); border-radius: 6px; height: 8px; overflow: hidden;">
                <div style="background: {border_color}; width: {conf_pct}%; height: 100%; border-radius: 6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation summary statement
    explanation = result.get("explanation", {})
    summary_text = explanation.get("summary", "")
    if summary_text:
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.7);
                border-left: 4px solid #3b82f6;
                padding: 14px 18px;
                border-radius: 0 8px 8px 0;
                margin-bottom: 20px;
                font-size: 15px;
                color: #e2e8f0;
                line-height: 1.6;
            ">
                <b>Explanation Summary:</b> {summary_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Claim analysis pills: Claim Type, Entities, Subclaims
    cols = st.columns([1, 2])
    with cols[0]:
        claim_type = explanation.get("claim_type", "FACTUAL")
        st.markdown(f"**Claim Type:** `{claim_type}`")
        entities = result.get("entities", [])
        if entities:
            st.markdown("**Entities Recognized:**")
            pills = " ".join([
                f"<span style='background: #1e293b; color: #38bdf8; border: 1px solid #0284c7; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px; display: inline-block;'>{e.get('text')} <small style='color: #94a3b8;'>({e.get('label')})</small></span>"
                for e in entities
            ])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No named entities detected.")

    with cols[1]:
        subclaims = result.get("subclaims", [])
        if subclaims:
            st.markdown(f"**Decomposed Propositions ({len(subclaims)}):**")
            for sc in subclaims:
                st.markdown(f"- 🧩 `{sc.get('text')}`")
        else:
            st.caption("Atomic claim (no decomposition required).")

    # Reasoning Steps Audit
    audit_steps = explanation.get("reasoning_steps", [])
    if audit_steps:
        with st.expander("🔍 Step-by-Step Reasoning Audit Trail", expanded=False):
            for step in audit_steps:
                st.markdown(f"- {step}")
