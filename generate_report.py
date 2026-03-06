import os
import argparse
import pandas as pd
from extract_text import extract_text
from dataset_prep import sent_tokenize, analyze_sentence
from predict_bias import load_model, RISK_MAPPING

def generate_bias_report(pdf_path: str, model_path: str, output_csv: str = None):
    """
    Reads a PDF, extracts all sentences, runs them through the trained model,
    and generates a CSV report showing all sentences flagged as High Risk.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find the file '{pdf_path}'")
        return
        
    print(f"--- Analyzing Document: {os.path.basename(pdf_path)} ---")
    
    # 1. Load the model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    # 2. Extract raw text from the document
    print("1. Extracting text...")
    try:
        raw_text = extract_text(pdf_path)
    except Exception as e:
        print(f"Failed to extract text: {e}")
        return
        
    # 3. Process sentences
    print("2. Tokenizing and analyzing sentences...")
    sentences = sent_tokenize(raw_text)
    
    report_data = []
    
    for i, sent_text in enumerate(sentences):
        sent_text = sent_text.strip()
        if len(sent_text) < 15: # Skip tiny artifacts
            continue
            
        # Extract features using NLTK
        features = analyze_sentence(sent_text)
        
        # Format for the model
        feature_list = [
            features["Word_Count"],
            features["Absolute_Count"],
            features["Subjective_Count"],
            features["Hedge_Count"],
            features["Hedge_Ratio"],
            features["First_Person_Count"]
        ]
        
        # Predict using the Random Forest model
        X_new = pd.DataFrame([feature_list], columns=['Word_Count', 'Absolute_Count', 'Subjective_Count', 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count'])
        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new)[0][prediction] * 100
        
        risk_category = RISK_MAPPING[prediction]
        
        # Add to our report if it's NOT Low Risk
        # You can change this to include Low Risk if you want the full document breakdown
        if risk_category in ("High Risk", "Medium Risk"):
            report_data.append({
                "Sentence_ID": i + 1,
                "Risk_Level": risk_category,
                "Confidence": f"{confidence:.1f}%",
                "Absolutes": features["Absolute_Count"],
                "Subjectives": features["Subjective_Count"],
                "Hedges": features["Hedge_Count"],
                "First_Person": features["First_Person_Count"],
                "Sentence_Text": sent_text
            })
            
    # 4. Generate the Summary Report
    df_report = pd.DataFrame(report_data)
    
    print("\n--- Bias Risk Analysis Complete ---")
    
    if df_report.empty:
        print(f"Good news! No Medium or High bias risk sentences were detected in {len(sentences)} total sentences.")
        return
        
    # Sort by risk level (High first) and then by confidence
    df_report['Risk_Sort_Val'] = df_report['Risk_Level'].map({"High Risk": 2, "Medium Risk": 1})
    df_report = df_report.sort_values(by=['Risk_Sort_Val', 'Confidence'], ascending=[False, False])
    df_report = df_report.drop(columns=['Risk_Sort_Val'])
    
    high_count = len(df_report[df_report['Risk_Level'] == 'High Risk'])
    med_count = len(df_report[df_report['Risk_Level'] == 'Medium Risk'])
    
    print(f"Total Sentences Analyzed: {len(sentences)}")
    print(f"High Risk Sentences Found: {high_count}")
    print(f"Medium Risk Sentences Found: {med_count}")
    
    # 5. Save the report to CSV
    if not output_csv:
        # Default save name based on the input document
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_csv = f"{base_name}_bias_report.csv"
        
    df_report.to_csv(output_csv, index=False)
    print(f"\n[+] Full report saved to: {os.path.abspath(output_csv)}")
    
    # Print a quick preview of the worst offenders
    if high_count > 0:
        print("\n--- Top High Risk Examples ---")
        top_high = df_report[df_report['Risk_Level'] == 'High Risk'].head(3)
        for _, row in top_high.iterrows():
            print(f"> [{row['Confidence']}] (Abs: {row['Absolutes']} | Subj: {row['Subjectives']}): {row['Sentence_Text'][:150]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Forensic PDF for cognitive bias risks.")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--output", "-o", help="Optional: Path to save the output CSV report")
    
    args = parser.parse_args()
    
    MODEL_FILE = r"c:\Users\Shambhavi\biasDetection\bias_model.pkl"
    
    output_target = args.output
    
    generate_bias_report(args.pdf_path, MODEL_FILE, output_csv=output_target)
