import glob
import os
import re
import string

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

from extract_text import extract_text

# --- 1. Domain-Specific Lexicons ---

EVIDENCE_TERMS = {
    "witness", "document", "recovery", "fsl", "report", "testimony", 
    "exhibit", "evidence", "deposition", "statement", "material", 
    "post-mortem", "medical", "expert", "pw", "dw", "panchnama", "seizure", "m.o."
}

REASONING_MARKERS = [
    "no material", "not proved", "in absence of", "proved beyond", "reasonable doubt",
    "established", "chain of", "fails to", "corroboration", "reliable", "trustworthy",
    "cogent", "consistent", "circumstantial", "inference", "burden of proof"
]

PROSECUTION_TERMS = {"prosecution", "complainant", "state", "informant", "police", "investigating"}
DEFENCE_TERMS = {"defence", "accused", "appellant", "respondent", "counsel"}

CONTRADICTION_TERMS = {
    "however", "but", "though", "although", "while", "nevertheless", "yet", 
    "despite", "contradiction", "omission", "discrepancy", "inconsistency", "contrary"
}

CERTAINTY_TERMS = {
    "clearly", "proved", "established", "must", "undoubtedly", "certainly", 
    "conclusively", "impossible", "irrefutable", "definite", "absolute", "entirely"
}

SUBJECTIVES = {
    "shocking", "terrible", "bizarre", "suspicious", "horrific", 
    "tragic", "disturbing", "unbelievable", "appalling", "ridiculous", 
    "gruesome", "vicious", "cruel", "brutal", "heinous"
}

# --- 2. Helper Functions ---

def is_legal_citation(text: str) -> bool:
    text = text.strip()
    if len(text) < 15:
        return True
    if 'indiankanoon.org' in text.lower() or re.match(r'^https?://', text, re.IGNORECASE):
        return True
    tokens = text.split()
    citation_patterns = sum(
        1 for tok in tokens if 
        re.match(r'^[\d]+[.,/\-]?[\d]*$', tok) or 
        re.match(r'^(s\.|sec\.|section|art\.|article|clause|order|rule|para|sub-section)', tok, re.IGNORECASE) or
        re.match(r'^[A-Z]{2,}[-/]\d', tok) or 
        re.match(r'^\(?[A-Z&]{2,5}\)?$', tok)
    )
    if not tokens:
        return True
    return (citation_patterns / len(tokens)) > 0.5

def strip_quoted_content(text: str) -> str:
    pattern = r'["\u201c\u201d\'][^"\u201c\u201d\']+["\u201c\u201d\']'
    return re.sub(pattern, ' ', text).strip()

def is_definition_sentence(text: str) -> bool:
    patterns = [r"^the term .* means", r"^definition of .* is", r"^.* shall mean", r"^expressions? .* includes?"]
    return any(re.search(p, text.lower()) for p in patterns)

# --- 3. Core Extraction ---

def analyze_sentence(text: str) -> dict:
    clean_text = strip_quoted_content(text)
    text_lower = clean_text.lower()
    
    # NLTK tokenization
    tokens = word_tokenize(clean_text)
    tokens_lower = [t.lower() for t in tokens if t not in string.punctuation]
    word_count = len(tokens_lower)
    
    # Feature counting
    evidence_count = sum(1 for t in tokens_lower if t in EVIDENCE_TERMS)
    reasoning_count = sum(1 for r in REASONING_MARKERS if r in text_lower)
    prosecution_count = sum(1 for t in tokens_lower if t in PROSECUTION_TERMS)
    defence_count = sum(1 for t in tokens_lower if t in DEFENCE_TERMS)
    contradiction_count = sum(1 for t in tokens_lower if t in CONTRADICTION_TERMS)
    certainty_count = sum(1 for t in tokens_lower if t in CERTAINTY_TERMS)
    subjective_count = sum(1 for t in tokens_lower if t in SUBJECTIVES)
    
    # Mathematical relationships
    certainty_support_ratio = certainty_count / (evidence_count + 1)
    
    # Heuristic Nature Tagging
    if evidence_count == 0 and reasoning_count == 0 and certainty_count > 0:
        nature = "Unsupported Conclusion"
    elif subjective_count > 0:
        nature = "Emotional Framing"
    elif reasoning_count > 0 or evidence_count > 0:
        if contradiction_count > 0:
            nature = "Critical Evaluation"
        else:
            nature = "Factual/Reasoning"
    elif prosecution_count > 0 or defence_count > 0:
        nature = "Argument Tracking"
    else:
        nature = "Neutral/Procedural"
        
    return {
        "Word_Count": word_count,
        "Evidence_Count": evidence_count,
        "Reasoning_Count": reasoning_count,
        "Prosecution_Mentions": prosecution_count,
        "Defence_Mentions": defence_count,
        "Contradiction_Count": contradiction_count,
        "Certainty_Count": certainty_count,
        "Subjective_Count": subjective_count,
        "Certainty_Support_Ratio": round(certainty_support_ratio, 2),
        "Statement_Nature": nature
    }

def get_contextual_metrics(window_features: list) -> dict:
    p_total = sum(f["Prosecution_Mentions"] for f in window_features)
    d_total = sum(f["Defence_Mentions"] for f in window_features)
    
    imbalance = abs(p_total - d_total) / (p_total + d_total + 1)
    
    return {
        "Contextual_Imbalance": round(imbalance, 2)
    }

def heuristic_bias_score(feat: dict, context: dict) -> int:
    """
    Simulates a human label for the sake of model training.
    Evaluates logical gaps rather than just vocabulary.
    """
    score = 0
    
    # Heavy penalty for emotional framing
    score += feat["Subjective_Count"] * 3
    
    # Penalty for high certainty claims made without supporting evidence
    if feat["Certainty_Support_Ratio"] >= 1.0 and feat["Certainty_Count"] > 0:
        score += 2
        
    # Penalty for highly imbalanced discussion context lacking critical evaluation
    if context["Contextual_Imbalance"] > 0.7:
        if feat["Contradiction_Count"] == 0 and feat["Reasoning_Count"] == 0:
            score += 1
            
    # Rewards / Neutralizers (Good Reasoning reduces perceived bias)
    if feat["Reasoning_Count"] > 0:
        score -= 1
    if feat["Contradiction_Count"] > 0:
        score -= 1
        
    if feat["Word_Count"] < 5:
        score = 0
        
    return max(0, score)

# --- 4. Pipeline Execution ---

def process_report(text: str, document_name: str) -> list:
    sentences = sent_tokenize(text)
    valid_sentences = [
        s.strip() for s in sentences 
        if len(s.strip()) >= 15 and not is_legal_citation(s) and not is_definition_sentence(s)
    ]
    
    analyzed_data = [analyze_sentence(s) for s in valid_sentences]
    
    results = []
    window_size = 3
    
    for i in range(len(valid_sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(valid_sentences), i + window_size // 2 + 1)
        window = analyzed_data[start:end]
        
        context_metrics = get_contextual_metrics(window)
        final_score = heuristic_bias_score(analyzed_data[i], context_metrics)
        
        # Determine bias label
        types = []
        if analyzed_data[i]["Certainty_Support_Ratio"] >= 1.0 and analyzed_data[i]["Certainty_Count"] > 0:
            types.append("Unsupported Conclusion / Assertion")
        if context_metrics["Contextual_Imbalance"] > 0.7 and analyzed_data[i]["Contradiction_Count"] == 0:
            types.append("Selective Evidence Focus")
        if analyzed_data[i]["Subjective_Count"] > 0:
            types.append("Subjective Framing")
            
        row = {
            "Document": document_name,
            "Sentence_ID": i + 1,
            "Sentence_Text": valid_sentences[i],
            "Context_Snippet": " | ".join(valid_sentences[start:end]),
            "Bias_Risk_Score": final_score,
            "Bias_Types": ", ".join(types) if types and final_score > 0 else "None",
        }
        row.update(analyzed_data[i])
        row.update(context_metrics)
        results.append(row)
        
    return results

def create_dataset(input_dir: str, output_csv: str):
    pattern = os.path.join(input_dir, "*.PDF")
    files = list(set(glob.glob(pattern) + glob.glob(pattern.lower())))
    
    all_data = []
    print(f"Processing {len(files)} files found in {input_dir}...")
    
    for count, file_path in enumerate(files, 1):
        base_name = os.path.basename(file_path)
        try:
            raw_text = extract_text(file_path)
            report_data = process_report(raw_text, base_name)
            all_data.extend(report_data)
            print(f"[{count}/{len(files)}] Processed: {base_name} ({len(report_data)} sentences)")
        except Exception as error:
            print(f"Skipping {base_name} due to error: {error}")
            
    if not all_data:
        print("No valid data points extracted.")
        return
        
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Successfully saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        create_dataset(sys.argv[1], sys.argv[2])
    else:
        # Defaults
        INPUT_PATH = "forensic_dataset"
        OUTPUT_PATH = "bias_dataset.csv"
        create_dataset(INPUT_PATH, OUTPUT_PATH)
