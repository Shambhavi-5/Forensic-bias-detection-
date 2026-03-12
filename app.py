import streamlit as st
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import our custom NLP Pipeline logic
from extract_text import extract_text
from dataset_prep import sent_tokenize, analyze_sentence, is_legal_citation, get_contextual_metrics
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
    """Runs the full NLP inference pipeline with contextual awareness."""
    
    # 1. Extract Text
    try:
        raw_text = extract_text(file_path)
    except Exception as e:
        st.error(f"Text extraction failed: {e}")
        return pd.DataFrame(), 0
        
    # 2. Tokenize and filter sentences
    raw_sentences = sent_tokenize(raw_text)
    valid_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= 15 and not is_legal_citation(s):
            valid_sentences.append(s)
    
    if not valid_sentences:
        return pd.DataFrame(), 0

    # 3. Step 1: Analyze ALL sentences for isolated features
    progress_bar = st.progress(0, text="Analyzing isolated linguistic features...")
    analyzed_list = []
    for i, sent in enumerate(valid_sentences):
        analyzed_list.append(analyze_sentence(sent))
        if i % 10 == 0:
            progress_bar.progress(min(i / len(valid_sentences), 1.0))

    # 4. Step 2: Contextual Analysis & ML Inference
    report_data = []
    total_sents = len(valid_sentences)
    window_size = 3
    
    progress_bar.progress(0, text="Calculating contextual bias density and momentum...")
    for i in range(total_sents):
        if i % 10 == 0:
            progress_bar.progress(min(i / total_sents, 1.0))
            
        # Define window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(total_sents, i + window_size // 2 + 1)
        window_features = analyzed_list[start_idx:end_idx]
        
        # Get contextual signals
        context_metrics = get_contextual_metrics(window_features)
        current_features = analyzed_list[i]
        
        # Combine all 9 features for the model
        feature_list = [
            current_features["Word_Count"],
            current_features["Absolute_Count"],
            current_features["Subjective_Count"],
            current_features["Hedge_Count"],
            current_features["Hedge_Ratio"],
            current_features["First_Person_Count"],
            context_metrics["Contextual_Density"],
            context_metrics["Contextual_Momentum"],
            current_features["Is_Opinion"]
        ]
        
        feature_columns = [
            'Word_Count', 'Absolute_Count', 'Subjective_Count', 
            'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count',
            'Contextual_Density', 'Contextual_Momentum', 'Is_Opinion'
        ]
        
        # ML Inference
        X_new = pd.DataFrame([feature_list], columns=feature_columns)
        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new)[0][prediction] * 100
        
        risk_category = RISK_MAPPING[prediction]
        
        if risk_category in ("High Risk", "Medium Risk"):
            snippet = " | ".join(valid_sentences[start_idx:end_idx])
            report_data.append({
                "Risk Level": risk_category,
                "Nature": current_features.get("Statement_Nature", "Fact"),
                "Confidence": f"{confidence:.1f}%",
                "Bias Types": current_features.get("Bias_Types", "—"),
                "Density": context_metrics["Contextual_Density"],
                "Key Triggers": current_features.get("Key_Triggers", "—"),
                "Sentence Text": valid_sentences[i],
                "Context Snippet": snippet
            })
            
    progress_bar.progress(1.0, text="Analysis complete!")
    
    df = pd.DataFrame(report_data)
    
    # Sort logically
    if not df.empty:
        df['Sort_Val'] = df['Risk Level'].map({"High Risk": 2, "Medium Risk": 1})
        df = df.sort_values(by=['Sort_Val', 'Confidence'], ascending=[False, False])
        df = df.drop(columns=['Sort_Val'])
        
    return df, len(raw_sentences)

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
            
            # Opinion Count
            opinion_count = len(df_results[df_results['Nature'] == 'Opinion'])
            col4.metric("📝 Subjective Opinions", opinion_count)
            
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
                height=500,
                column_config={
                    "Risk Level": st.column_config.TextColumn("Risk", width="small"),
                    "Nature": st.column_config.TextColumn("Nature", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Bias Types": st.column_config.TextColumn("Bias Types", width="medium"),
                    "Key Triggers": st.column_config.TextColumn("Key Triggers", width="medium"),
                    "Sentence Text": st.column_config.TextColumn("Sentence Text", width="large"),
                    "Context Snippet": st.column_config.TextColumn("Context Snippet", width="large")
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
