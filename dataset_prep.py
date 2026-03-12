import os
import re
import glob
import pandas as pd
import string
from extract_text import extract_text

# Import NLTK components
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# --- Lexicons for Target Features ---
# 1. Absolute / Overconfident Terms
ABSOLUTES = {
    "always", "never", "undoubtedly", "obviously", "certainly", 
    "conclusively", "impossible", "clearly", "unquestionably", 
    "irrefutable", "definite", "must", "proven", "proves", 
    "unanimously", "absolute", "entirely", "completely"
}

# 2. Emotional / Subjective Adjectives & Adverbs
SUBJECTIVES = {
    "shocking", "terrible", "bizarre", "suspicious", "horrific", 
    "tragic", "disturbing", "unbelievable", "appalling", "ridiculous",
    "fortunately", "unfortunately", "sadly", "luckily", "surprisingly",
    "gruesome", "vicious", "cruel", "brutal", "heinous"
}

# 3. Hedging / Uncertainty Terms
HEDGES = {
    "might", "may", "could", "suggests", "appears", "seems", "possibly", 
    "probably", "likely", "unlikely", "indicates", "implies", "perhaps",
    "tends", "assumes", "believes", "estimates" , "maybe"
}

def is_legal_citation(sentence_text: str) -> bool:
    """
    Returns True if a sentence is predominantly a legal citation, URL,
    section/law reference, or case number — not real prose worth analyzing.
    """
    text = sentence_text.strip()
    
    # Reject very short fragments
    if len(text) < 15:
        return True

    # Reject sentences that are just URLs
    if re.match(r'^https?://', text, re.IGNORECASE):
        return True

    # Reject Indian Kanoon / URL patterns embedded in text
    if 'indiankanoon.org' in text.lower():
        return True

    # Tokenize into words to check composition
    tokens = text.split()
    total = len(tokens)
    if total == 0:
        return True

    # Count tokens that look like: numbers, section refs, case IDs, legal codes
    citation_tokens = 0
    for tok in tokens:
        # Pure numbers or numbers with punctuation  (e.g. "168", "421379")
        if re.match(r'^[\d]+[.,/\-]?[\d]*$', tok):
            citation_tokens += 1
        # Section / article / clause references (e.g. "65B(4)", "S.168", "Art.21")
        elif re.match(r'^(s\.|sec\.|section|art\.|article|clause|order|rule|para|sub-section)', tok, re.IGNORECASE):
            citation_tokens += 1
        # Case number patterns like "No.2/2024", "FAO-6160"
        elif re.match(r'^[A-Z]{2,}[-/]\d', tok):
            citation_tokens += 1
        # Legal doc identifiers (e.g. "O&M", "DB", "SLP")
        elif re.match(r'^\(?[A-Z&]{2,5}\)?$', tok):
            citation_tokens += 1

    # If more than 50% of the tokens look like citation fragments, skip it
    if total > 0 and (citation_tokens / total) > 0.5:
        return True

    return False


def get_bias_types(absolute_count: int, subjective_count: int, hedge_count: int, absolute_words: list, subjective_words: list) -> list:
    """
    Returns a list of specific bias type labels based on detected features.
    - Overconfidence Bias: high use of absolute/definitive language with low hedging
    - Emotional Framing Bias: use of charged, subjective adjectives
    - Confirmation Bias (Anchoring): statement presented as fact without hedge
    """
    bias_types = []

    if absolute_count > 0 and hedge_count == 0:
        bias_types.append("Overconfidence Bias")
    
    if absolute_count > 0 and hedge_count > 0 and absolute_count > hedge_count:
        bias_types.append("Anchoring Bias")

    if subjective_count > 0:
        bias_types.append("Emotional Framing Bias")

    if absolute_count >= 2 and subjective_count == 0 and hedge_count == 0:
        # Multiple absolutes with no subjectivity or hedging = confirmation bias pattern
        if "Overconfidence Bias" not in bias_types:
            bias_types.append("Confirmation Bias")

    return bias_types if bias_types else ["Unspecified Risk"]


def analyze_sentence(sentence_text):
    """
    Analyzes a single sentence string and extracts linguistic features
    associated with cognitive bias risks using NLTK.
    """
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence_text)
    
    absolute_count = 0
    subjective_count = 0
    hedge_count = 0
    first_person_count = 0
    word_count = 0
    matched_absolutes = []
    matched_subjectives = []
    
    # Tag parts of speech
    tagged_tokens = pos_tag(tokens)
    
    for token, pos in tagged_tokens:
        # Skip punctuation
        if token in string.punctuation:
            continue
            
        word_lower = token.lower()
        word_count += 1
        
        # Check against lexicons
        if word_lower in ABSOLUTES:
            absolute_count += 1
            matched_absolutes.append(word_lower)
        elif word_lower in SUBJECTIVES:
            subjective_count += 1
            matched_subjectives.append(word_lower)
        elif word_lower in HEDGES:
            hedge_count += 1
            
        # First-person pronouns (I, me, my, mine, we, us, our) — tracked but NOT scored
        # (standard legal language uses "I have heard..." which is procedural, not bias)
        if pos in ("PRP", "PRP$") and word_lower in {"i", "me", "my", "mine", "we", "us", "our"}:
            first_person_count += 1

    # Calculate Hedge Ratio (Hedges per Absolute)
    if absolute_count > 0:
        hedge_ratio = hedge_count / absolute_count
    else:
        hedge_ratio = hedge_count
        
    # Bias Risk Score: Only penalize Absolutes and Subjectives
    # First-person is tracked but NOT scored (too common in legal procedural language)
    bias_score = (absolute_count * 2) + (subjective_count * 3)

    # Determine specific bias types from the features
    bias_types = get_bias_types(absolute_count, subjective_count, hedge_count, matched_absolutes, matched_subjectives)
        
    return {
        "Word_Count": word_count,
        "Absolute_Count": absolute_count,
        "Subjective_Count": subjective_count,
        "Hedge_Count": hedge_count,
        "Hedge_Ratio": round(hedge_ratio, 2),
        "First_Person_Count": first_person_count,
        "Bias_Risk_Score": bias_score,
        "Bias_Types": ", ".join(bias_types)
    }

def process_report(text: str, document_name: str) -> list:
    """
    Processes the full text of a report and returns a list of dictionaries,
    one for each sentence.
    """
    # Tokenize text into sentences using NLTK
    sentences = sent_tokenize(text)
    
    sentence_data = []
    
    for i, sent_text in enumerate(sentences):
        sent_text = sent_text.strip()
        
        # Skip empty, tiny, or legal citation sentences
        if len(sent_text) < 15 or is_legal_citation(sent_text):
            continue
            
        features = analyze_sentence(sent_text)
        
        # Combine base info and linguistic features
        row = {
            "Document": document_name,
            "Sentence_ID": i + 1,
            "Sentence_Text": sent_text
        }
        row.update(features)
        
        sentence_data.append(row)
        
    return sentence_data

def create_dataset(input_dir: str, output_csv: str):
    """
    Iterates through all supported files in a directory, extracts the text,
    analyzes it, and saves the resulting dataset to a CSV file.
    """
    print(f"Starting dataset creation from: {input_dir}")
    print(f"Output will be saved to: {output_csv}")
    
    all_sentences = []
    
    # Find all PDFs in the directory (case insensitive via multiple globs if needed)
    search_pattern_upper = os.path.join(input_dir, "*.PDF")
    search_pattern_lower = os.path.join(input_dir, "*.pdf")
    
    file_paths = glob.glob(search_pattern_upper) + glob.glob(search_pattern_lower)
    # Remove duplicates if any
    file_paths = list(set(file_paths)) 
    
    print(f"Found {len(file_paths)} PDF reports to process.")
    
    for fp in file_paths:
        filename = os.path.basename(fp)
        print(f"Processing: {filename}...")
        
        try:
            # Stage 1: Text Extraction (using the module built previously)
            raw_text = extract_text(fp)
            
            # Stage 2: NLP Processing
            doc_sentences = process_report(raw_text, filename)
            
            all_sentences.extend(doc_sentences)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_sentences)
    
    if df.empty:
        print("No valid sentences extracted! Check if the PDFs contain readable text.")
        return
        
    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n--- SUCCESS ---")
    print(f"Extracted {len(df)} sentences across {len(file_paths)} reports.")
    print(f"Dataset saved successfully!")

if __name__ == "__main__":
    # The input directory provided by the user containing the sample dataset
    INPUT_DIR = r"c:\Users\Shambhavi\Desktop\forensic_dataset (1)"
    
    # The target output file
    OUTPUT_FILE = r"c:\Users\Shambhavi\biasDetection\bias_dataset.csv"
    
    # Run the dataset generation
    create_dataset(INPUT_DIR, OUTPUT_FILE)
