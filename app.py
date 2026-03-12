import streamlit as st
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import our custom NLP Pipeline logic
from extract_text import extract_text
from dataset_prep import sent_tokenize, analyze_sentence, is_legal_citation
from predict_bias import load_model, RISK_MAPPING

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Forensic Bias Detector",
    page_icon="⚖️",
    layout="wide"
)

MODEL_PATH = r"c:\Users\Shambhavi\biasDetection\bias_model.pkl"

@st.cache_resource
def get_model():
    """Loads the model once and caches it in memory for performance."""
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load the model. Ensure `model_training.py` has been run.\nError: {e}")
        return None

def process_document(file_path: str, model):
    """Runs the full NLP inference pipeline on a document and returns the dataframe."""
    
    # 1. Extract Text
    try:
        raw_text = extract_text(file_path)
    except Exception as e:
        st.error(f"Text extraction failed: {e}")
        return pd.DataFrame(), 0
        
    # 2. Tokenize into sentences
    sentences = sent_tokenize(raw_text)
    report_data = []
    
    # 3. Analyze each sentence
    progress_bar = st.progress(0)
    total_sents = len(sentences)
    
    for i, sent_text in enumerate(sentences):
        # Update progress bar
        if i % 10 == 0:
            progress_bar.progress(min(i / total_sents, 1.0))
            
        sent_text = sent_text.strip()
        # Skip fragments and legal citation sentences (URLs, section refs, case numbers)
        if len(sent_text) < 15 or is_legal_citation(sent_text):
            continue
            
        # Extract Linguistic Features
        features = analyze_sentence(sent_text)
        
        feature_list = [
            features["Word_Count"],
            features["Absolute_Count"],
            features["Subjective_Count"],
            features["Hedge_Count"],
            features["Hedge_Ratio"],
            features["First_Person_Count"]
        ]
        
        # ML Inference
        X_new = pd.DataFrame([feature_list], columns=['Word_Count', 'Absolute_Count', 'Subjective_Count', 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count'])
        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new)[0][prediction] * 100
        
        risk_category = RISK_MAPPING[prediction]
        
        # Only log Medium & High risk items to avoid cluttering the UI
        if risk_category in ("High Risk", "Medium Risk"):
            report_data.append({
                "Risk Level": risk_category,
                "Confidence": f"{confidence:.1f}%",
                "Bias Types": features.get("Bias_Types", "—"),
                "Absolutes": features["Absolute_Count"],
                "Subjectives": features["Subjective_Count"],
                "Sentence Text": sent_text
            })
            
    progress_bar.progress(1.0)
    
    df = pd.DataFrame(report_data)
    
    # Sort logically
    if not df.empty:
        df['Sort_Val'] = df['Risk Level'].map({"High Risk": 2, "Medium Risk": 1})
        df = df.sort_values(by=['Sort_Val', 'Confidence'], ascending=[False, False])
        df = df.drop(columns=['Sort_Val'])
        
    return df, total_sents

# --- UI LAYOUT ---
st.title("⚖️ Cognitive Bias Risk Detector")
st.markdown("""
Welcome to the MVP for AI-Assisted Cognitive Bias Risk Detection in Forensic Reports. 
Upload a PDF forensic report to instantly identify subjective framing, overconfidence, and confirmation bias indicators.
""")

st.divider()

# Load Model
model = get_model()

if model:
    uploaded_file = st.file_uploader("Upload a Forensic Report (.pdf or .docx)", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        
        # Save uploaded file temporarily so our extraction script can read it from disk
        with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        st.info("Analyzing Document... Please wait.")
        
        # Run Pipeline
        df_results, total_sentences = process_document(tmp_file_path, model)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if df_results.empty:
            st.success(f"✅ Analysis complete! {total_sentences} sentences scanned. No high/medium bias risks were detected.")
        else:
            # Calculate metrics
            high_count = len(df_results[df_results['Risk Level'] == 'High Risk'])
            med_count = len(df_results[df_results['Risk Level'] == 'Medium Risk'])
            
            # Display Metrics
            st.subheader("📊 Document Risk Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Sentences Analyzed", total_sentences)
            col2.metric("🔴 High Risk Sentences", high_count)
            col3.metric("🟠 Medium Risk Sentences", med_count)
            
            # Bias type breakdown
            if "Bias Types" in df_results.columns:
                bias_counts = df_results["Bias Types"].str.split(", ").explode().value_counts()
                top_bias = bias_counts.idxmax() if not bias_counts.empty else "None"
                col4.metric("Top Bias Type", top_bias)
            
            st.divider()
            
            # Display Interactive Data Table
            st.subheader("🚩 Flagged Sentences")
            
            # Custom styling function for pandas
            def color_risk(val):
                if val == 'High Risk':
                    return 'color: red; font-weight: bold'
                elif val == 'Medium Risk':
                    return 'color: orange'
                return ''
                
            styled_df = df_results.style.map(color_risk, subset=['Risk Level'])
            
            st.dataframe(
                styled_df, 
                use_container_width=True,
                height=450,
                column_config={
                    "Bias Types": st.column_config.TextColumn("Bias Types", width="medium"),
                    "Sentence Text": st.column_config.TextColumn("Sentence Text", width="large")
                }
            )
            
            # CSV Download Button
            st.markdown("### Export Results")
            csv = df_results.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Full Bias Report (CSV)",
                data=csv,
                file_name=f"{uploaded_file.name}_bias_report.csv",
                mime="text/csv",
            )
