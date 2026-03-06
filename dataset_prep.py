import os
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
    "tends", "assumes", "believes", "estimates"
}

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
        elif word_lower in SUBJECTIVES:
            subjective_count += 1
        elif word_lower in HEDGES:
            hedge_count += 1
            
        # First-person pronouns (I, me, my, mine, we, us, our)
        # PRP = Personal Pronoun, PRP$ = Possessive Pronoun
        if pos in ("PRP", "PRP$") and word_lower in {"i", "me", "my", "mine", "we", "us", "our"}:
            first_person_count += 1

    # Calculate Hedge Ratio (Hedges per Absolute)
    # A ratio < 1 means they use more definitives than hedges (overconfidence)
    # A ratio > 1 means they are hedging heavily
    if absolute_count > 0:
        hedge_ratio = hedge_count / absolute_count
    else:
        hedge_ratio = hedge_count # Treat 0 absolutes as a denominator of 1 for the ratio
        
    # Total Bias Indicators Score (A simple aggregate for the MVP)
    # We heavily penalize Subjectives and Absolutes. 
    bias_score = (absolute_count * 2) + (subjective_count * 3) + first_person_count
        
    return {
        "Word_Count": word_count,
        "Absolute_Count": absolute_count,
        "Subjective_Count": subjective_count,
        "Hedge_Count": hedge_count,
        "Hedge_Ratio": round(hedge_ratio, 2),
        "First_Person_Count": first_person_count,
        "Bias_Risk_Score": bias_score
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
        
        # Skip empty or tiny sentences (e.g. just a number, abbreviation, or artifact)
        if len(sent_text) < 15:
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
