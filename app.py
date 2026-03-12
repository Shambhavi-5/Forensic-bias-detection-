import streamlit as st
import pandas as pd
import os
from tempfile import NamedTemporaryFile
from extract_text import extract_text
from dataset_prep import sent_tokenize, analyze_sentence, is_legal_citation, get_contextual_metrics
from predict_bias import load_model, RISK_MAPPING

st.set_page_config(page_title="Forensic Bias Detector", page_icon="⚖️", layout="wide")
MODEL_PATH = "bias_model.pkl"

@st.cache_resource
def get_model():
    try: return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load the model. Run `model_training.py` first.\nError: {e}"); return None

def process_document(file_path, model):
    try: raw_text = extract_text(file_path)
    except Exception as e: return st.error(f"Text extraction failed: {e}"), (pd.DataFrame(), 0)
    
    raw_sents = sent_tokenize(raw_text)
    valid_sents = [s.strip() for s in raw_sents if len(s.strip()) >= 15 and not is_legal_citation(s)]
    if not valid_sents: return pd.DataFrame(), 0

    p_bar = st.progress(0, text="Analyzing features...")
    analyzed = [analyze_sentence(s) for s in valid_sents]
    
    report_data, win = [], 3
    for i in range(len(valid_sents)):
        if i % 10 == 0: p_bar.progress(i / len(valid_sents), text="Calculating contextual bias...")
        ctx = get_contextual_metrics(analyzed[max(0, i-win//2) : min(len(valid_sents), i+win//2+1)])
        cur = analyzed[i]
        
        feat_cols = ['Word_Count', 'Absolute_Count', 'Subjective_Count', 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count', 'Contextual_Density', 'Contextual_Momentum', 'Is_Opinion']
        feats = [cur["Word_Count"], cur["Absolute_Count"], cur["Subjective_Count"], cur["Hedge_Count"], cur["Hedge_Ratio"], cur["First_Person_Count"], ctx["Contextual_Density"], ctx["Contextual_Momentum"], cur["Is_Opinion"]]
        
        pred = model.predict(pd.DataFrame([feats], columns=feat_cols))[0]
        risk = RISK_MAPPING[pred]
        
        if risk in ("High Risk", "Medium Risk"):
            report_data.append({"Risk Level": risk, "Confidence": f"{model.predict_proba(pd.DataFrame([feats], columns=feat_cols))[0][pred]*100:.1f}%",
                                "Bias Types": cur.get("Bias_Types", "—"), "Density": ctx["Contextual_Density"],
                                "Key Triggers": cur.get("Key_Triggers", "—"), "Sentence Text": valid_sents[i],
                                "Context Snippet": " | ".join(valid_sents[max(0, i-win//2) : min(len(valid_sents), i+win//2+1)])})
            
    p_bar.progress(1.0, text="Analysis complete!")
    df = pd.DataFrame(report_data)
    if not df.empty:
        df['Sort'] = df['Risk Level'].map({"High Risk": 2, "Medium Risk": 1})
        df = df.sort_values(by=['Sort', 'Confidence'], ascending=[False, False]).drop(columns=['Sort'])
    return df, len(raw_sents)

st.title("⚖️ Cognitive Bias Risk Detector")
st.markdown("Welcome to the MVP for AI-Assisted Cognitive Bias Risk Detection. Upload a forensic report to identify bias indicators.")
st.divider()

if model := get_model():
    if uploaded_file := st.file_uploader("Upload a Forensic Report (.pdf or .docx)", type=["pdf", "docx"]):
        with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
        
        st.info("Analyzing Document...")
        df_res, total_sents = process_document(tmp_path, model); os.unlink(tmp_path)
        
        if df_res.empty: st.success(f"✅ {total_sents} sentences scanned. No risks detected.")
        else:
            st.subheader("📊 Document Risk Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Sentences", total_sents)
            c2.metric("🔴 High Risk", len(df_res[df_res['Risk Level'] == 'High Risk']))
            c3.metric("🟠 Medium Risk", len(df_res[df_res['Risk Level'] == 'Medium Risk']))
            
            top_bias = df_res["Bias Types"].str.split(", ").explode().value_counts().idxmax() if "Bias Types" in df_res.columns else "None"
            c4.metric("Top Bias", top_bias)
            
            st.divider(); st.subheader("🚩 Flagged Sentences")
            st.dataframe(df_res.style.map(lambda v: 'color: red; font-weight: bold' if v == 'High Risk' else 'color: orange' if v == 'Medium Risk' else '', subset=['Risk Level']),
                         use_container_width=True, height=500,
                         column_config={"Risk Level": st.column_config.TextColumn("Risk", width="small"),
                                        "Confidence": st.column_config.TextColumn("Conf.", width="small"), "Sentence Text": st.column_config.TextColumn("Text", width="large")})
            
            st.markdown("### Export Results")
            st.download_button("📥 Download Bias Report (CSV)", df_res.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_bias_report.csv", "text/csv")
