"""
app.py — Streamlit UI for Legal Bias & Reasoning Flaw Detection.
Wraps the NLI-powered pipeline (main.py) in a rich interactive dashboard.
"""

import streamlit as st
import pandas as pd
import os
from tempfile import NamedTemporaryFile

from main import run_pipeline, CLAIM_LABELS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Reasoning & Bias Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/judge.png", width=80)
    st.title("⚖️ Legal Bias Detector")
    st.markdown(
        "**NLI-powered pipeline** for detecting logical flaws and bias in legal & forensic documents.\n\n"
        "_Not keyword-based. Uses semantic NLI reasoning._"
    )
    st.divider()
    top_k = st.slider("Evidence sentences per claim (top-k)", min_value=1, max_value=8, value=3, step=1)
    show_non_claims = st.checkbox("Show Background / Evidence rows", value=False)
    st.divider()
    st.caption("Architecture: pdf_parser → SentenceTransformer → Zero-Shot Classifier → Evidence Retriever → NLI Verifier → Reasoning Engine")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; padding-bottom:0;'>⚖️ Legal Reasoning & Bias Detector</h1>
    <p style='text-align:center; color:gray; margin-top:4px; font-size:15px;'>
        Semantic NLI-powered analysis of judicial and forensic documents
    </p>
""", unsafe_allow_html=True)
st.divider()

# ─── Risk color helpers ────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Potentially Biased / Flawed":        "#c0392b",
    "Weakly Supported (Overconfident)":   "#e67e22",
    "Weakly Supported":                   "#f39c12",
    "Inconclusive":                        "#8e44ad",
    "Well-Reasoned":                       "#27ae60",
    "Background / Evidence":              "#2980b9",
}

def badge(label):
    color = LABEL_COLORS.get(label, "#555")
    return f"<span style='background:{color};color:white;border-radius:5px;padding:2px 8px;font-size:12px;font-weight:600'>{label}</span>"

def row_bgcolor(label):
    colors = {
        "Potentially Biased / Flawed":        "rgba(192,57,43,0.10)",
        "Weakly Supported (Overconfident)":   "rgba(230,126,34,0.10)",
        "Weakly Supported":                   "rgba(243,156,18,0.08)",
        "Inconclusive":                        "rgba(142,68,173,0.08)",
        "Well-Reasoned":                       "rgba(39,174,96,0.08)",
        "Background / Evidence":              "",
    }
    return colors.get(label, "")

def style_df(df: pd.DataFrame):
    def highlight(row):
        color = row_bgcolor(row["Final_Label"])
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(highlight, axis=1)

# ─── Upload & Analyze ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📄 Upload a court judgment, order, or forensic report",
    type=["pdf", "docx", "txt"],
)

if uploaded:
    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.info(f"📂 Processing **{uploaded.name}** — Loading models if first run (may take a moment)…")

    with st.spinner("Running NLI pipeline…"):
        try:
            df = run_pipeline(
                file_path=tmp_path,
                top_k=top_k,
                output_csv=None,
                verbose=False,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            df = pd.DataFrame()
        finally:
            os.unlink(tmp_path)

    if df.empty:
        st.error("No sentences could be extracted. Please check the document format.")
    else:
        # ─── Metrics ──────────────────────────────────────────────────────────
        claims_df   = df[df["Sentence_Type"].isin(CLAIM_LABELS)]
        high_risk   = df[df["Final_Label"].str.contains("Biased|Overconfident", na=False)]
        weak        = df[df["Final_Label"] == "Weakly Supported"]
        well        = df[df["Final_Label"] == "Well-Reasoned"]

        st.subheader("📊 Document Intelligence Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sentences",         len(df))
        c2.metric("Claims / Arguments",       len(claims_df))
        c3.metric("🔴 High Risk",              len(high_risk))
        c4.metric("🟡 Weakly Supported",       len(weak))
        c5.metric("🟢 Well-Reasoned",          len(well))
        st.divider()

        # ─── Risk Distribution bar ────────────────────────────────────────────
        st.subheader("📈 Risk Label Distribution")
        label_counts = df["Final_Label"].value_counts()
        st.bar_chart(label_counts)
        st.divider()

        # ─── Filtered table ───────────────────────────────────────────────────
        st.subheader("🔍 Sentence-level Analysis")
        display_df = df if show_non_claims else df[df["Final_Label"] != "Background / Evidence"]

        # Shorten retrieved evidence for display
        display_df = display_df.copy()
        display_df["Retrieved_Evidence"] = display_df["Retrieved_Evidence"].str[:250]

        styled = style_df(display_df)
        st.dataframe(
            styled,
            use_container_width=True,
            height=520,
            column_config={
                "Sentence":               st.column_config.TextColumn("Sentence",          width="large"),
                "Sentence_Type":          st.column_config.TextColumn("Type",              width="medium"),
                "NLI_Verdict":            st.column_config.TextColumn("NLI Verdict",       width="medium"),
                "NLI_Score":              st.column_config.NumberColumn("NLI Score",       format="%.3f"),
                "Bias_Risk_Score":        st.column_config.NumberColumn("Bias Risk ↑",     format="%.2f"),
                "Final_Label":            st.column_config.TextColumn("Verdict",           width="medium"),
                "Explanation":            st.column_config.TextColumn("Explanation",       width="large"),
                "Classifier_Confidence":  st.column_config.NumberColumn("Classif. Conf.", format="%.3f"),
                "Retrieved_Evidence":     st.column_config.TextColumn("Evidence Used",     width="large"),
            },
        )
        st.divider()

        # ─── Detailed Expandable Cards for High Risk ──────────────────────────
        if not high_risk.empty:
            st.subheader("🚨 High-Risk Claims — Detailed Review")
            for _, row in high_risk.iterrows():
                label_html = badge(row["Final_Label"])
                with st.expander(f"{row['Sentence'][:100]}…", expanded=False):
                    st.markdown(f"**Verdict:** {label_html}", unsafe_allow_html=True)
                    st.markdown(f"**Bias Risk Score:** `{row['Bias_Risk_Score']}` &nbsp;|&nbsp; **NLI:** `{row['NLI_Verdict']}` ({row['NLI_Score']})", unsafe_allow_html=True)
                    st.markdown(f"**Explanation:** {row['Explanation']}")
                    if row.get("Retrieved_Evidence") and row["Retrieved_Evidence"] != "N/A":
                        st.markdown("**Key Evidence Considered:**")
                        for ev_chunk in row["Retrieved_Evidence"].split(" | "):
                            if ev_chunk.strip():
                                st.caption(f"• {ev_chunk.strip()}")
            st.divider()

        # ─── Export ───────────────────────────────────────────────────────────
        st.markdown("### 📥 Export Full Report")
        csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"{uploaded.name}_bias_report.csv",
            mime="text/csv",
        )

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#888;'>
        <div style='font-size:64px;'>⚖️</div>
        <h3>Upload a legal document to begin analysis</h3>
        <p>Supports PDF, DOCX, and TXT formats.<br/>
        The pipeline uses Sentence Transformers + NLI models to semantically evaluate every claim.</p>
    </div>
    """, unsafe_allow_html=True)