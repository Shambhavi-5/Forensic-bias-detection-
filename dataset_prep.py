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
    "tends", "assumes", "believes", "estimates", "maybe"
}

# --- Edge Case Helper Patterns ---

# Reported speech: bias words belong to prosecution/counsel/complainant, not the author
REPORTED_SPEECH_VERBS = re.compile(
    r'\b(argued|submitted|contended|alleged|claimed|stated|held|opined|observed|said|urged|insisted)'
    r'\s+that\b',
    re.IGNORECASE
)

# Conditional / hypothetical sentence starters
CONDITIONAL_STARTERS = re.compile(
    r'^\s*(if|had|were|should|supposing|assuming|hypothetically|provided that|in the event)\b',
    re.IGNORECASE
)

# Negation words that, if they immediately precede a bias term, cancel it out
NEGATION_WORDS = {
    "not", "no", "never", "nor", "neither", "cannot", "doesn't", "don't",
    "does", "didn't", "wasn't", "isn't", "aren't", "couldn't", "wouldn't", "shouldn't"
}


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def is_legal_citation(sentence_text: str) -> bool:
    """
    Returns True if a sentence is predominantly a legal citation, URL,
    section/law reference, or case number — not real prose worth analyzing.
    """
    text = sentence_text.strip()

    if len(text) < 15:
        return True

    # Reject bare URLs
    if re.match(r'^https?://', text, re.IGNORECASE):
        return True

    # Reject Indian Kanoon / URL patterns embedded in text
    if 'indiankanoon.org' in text.lower():
        return True

    tokens = text.split()
    total = len(tokens)
    if total == 0:
        return True

    citation_tokens = 0
    for tok in tokens:
        # Pure numbers or numbers with punctuation (e.g. "168", "421379")
        if re.match(r'^[\d]+[.,/\-]?[\d]*$', tok):
            citation_tokens += 1
        # Section / article / clause references
        elif re.match(r'^(s\.|sec\.|section|art\.|article|clause|order|rule|para|sub-section)', tok, re.IGNORECASE):
            citation_tokens += 1
        # Case number patterns like "No.2/2024", "FAO-6160"
        elif re.match(r'^[A-Z]{2,}[-/]\d', tok):
            citation_tokens += 1
        # Legal doc identifiers (e.g. "O&M", "DB", "SLP")
        elif re.match(r'^\(?[A-Z&]{2,5}\)?$', tok):
            citation_tokens += 1

    # If more than 50% of tokens look like citation fragments, skip it
    if (citation_tokens / total) > 0.5:
        return True

    return False


def strip_quoted_content(text: str) -> str:
    """
    Removes content inside double quotation marks from a sentence before analysis.
    Words inside quotes belong to a court ruling, a witness, or a party —
    not the report author — so they should not be scored for bias.
    Handles both ASCII double quotes and Unicode curly quotes.
    """
    stripped = re.sub(r'["\u201c\u201d][^"\u201c\u201d]*["\u201c\u201d]', ' ', text)
    return stripped.strip()


def get_context_penalty(sentence_text: str) -> float:
    """
    Returns a multiplier (0.0–1.0) to scale the bias score based on context:
      0.0 = completely exclude (conditionals and hypotheticals)
      0.5 = halve the score (reported/attributed speech)
      1.0 = full score (direct author assertion)
    """
    # Conditionals: the claim is not being asserted as a real fact
    if CONDITIONAL_STARTERS.match(sentence_text):
        return 0.0

    # Reported speech: the bias belongs to a third party being quoted
    if REPORTED_SPEECH_VERBS.search(sentence_text):
        return 0.5

    return 1.0


# =============================================================================
# BIAS TYPE CLASSIFIER
# =============================================================================

def get_bias_types(absolute_count: int, subjective_count: int, hedge_count: int,
                   absolute_words: list, subjective_words: list) -> list:
    """
    Returns a list of specific bias type labels based on detected features.
    """
    bias_types = []

    if absolute_count > 0 and hedge_count == 0:
        bias_types.append("Overconfidence Bias")

    if absolute_count > 0 and hedge_count > 0 and absolute_count > hedge_count:
        bias_types.append("Anchoring Bias")

    if subjective_count > 0:
        bias_types.append("Emotional Framing Bias")

    if absolute_count >= 2 and subjective_count == 0 and hedge_count == 0:
        if "Overconfidence Bias" not in bias_types:
            bias_types.append("Confirmation Bias")

    return bias_types if bias_types else ["Unspecified Risk"]


# =============================================================================
# MAIN SENTENCE ANALYZER
# =============================================================================

def analyze_sentence(sentence_text: str) -> dict:
    """
    Analyzes a single sentence and extracts linguistic features
    associated with cognitive bias risks using NLTK.

    Edge cases handled:
      1. Quoted text   → stripped before analysis (court holdings, witness quotes)
      2. Negated terms → 'does not conclusively prove' does not count
      3. Reported speech → 'counsel argued clearly...' score is halved
      4. Conditionals  → 'if proven conclusively...' scores 0
    """
    # Step 1: Compute context penalty using the FULL original sentence
    context_penalty = get_context_penalty(sentence_text)

    # Step 2: Strip quoted content before counting bias words
    analysis_text = strip_quoted_content(sentence_text)

    # Step 3: Tokenize and POS-tag the cleaned text
    tokens = word_tokenize(analysis_text)
    tagged_tokens = pos_tag(tokens)

    absolute_count = 0
    subjective_count = 0
    hedge_count = 0
    first_person_count = 0
    word_count = 0
    matched_absolutes = []
    matched_subjectives = []

    for idx, (token, pos) in enumerate(tagged_tokens):
        if token in string.punctuation:
            continue

        word_lower = token.lower()
        word_count += 1

        # Step 4: Negation check — look back 1-2 tokens for negation words
        preceding_words = [
            tagged_tokens[i][0].lower()
            for i in range(max(0, idx - 2), idx)
            if tagged_tokens[i][0] not in string.punctuation
        ]
        is_negated = any(w in NEGATION_WORDS for w in preceding_words)

        # Check against lexicons — skip if this word is negated
        if word_lower in ABSOLUTES and not is_negated:
            absolute_count += 1
            matched_absolutes.append(word_lower)
        elif word_lower in SUBJECTIVES and not is_negated:
            subjective_count += 1
            matched_subjectives.append(word_lower)
        elif word_lower in HEDGES:
            hedge_count += 1

        # First-person pronouns: tracked but NOT scored
        # (legal procedural language routinely uses "I have heard...", "In my opinion...")
        if pos in ("PRP", "PRP$") and word_lower in {"i", "me", "my", "mine", "we", "us", "our"}:
            first_person_count += 1

    # Step 5: Calculate Hedge Ratio
    hedge_ratio = (hedge_count / absolute_count) if absolute_count > 0 else hedge_count

    # Step 6: Apply context penalty to final score
    raw_score = (absolute_count * 2) + (subjective_count * 3)
    bias_score = round(raw_score * context_penalty)

    # Step 7: Label the bias types; annotate if context reduced the score
    bias_types = get_bias_types(absolute_count, subjective_count, hedge_count,
                                matched_absolutes, matched_subjectives)
    if context_penalty == 0.0 and raw_score > 0:
        bias_types = ["Conditional/Hypothetical Language"]
    elif context_penalty == 0.5 and raw_score > 0:
        bias_types = [b + " (Attributed)" for b in bias_types]

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


# =============================================================================
# DATASET GENERATION
# =============================================================================

def process_report(text: str, document_name: str) -> list:
    """
    Processes the full text of one report.
    Returns a list of dicts, one per valid sentence.
    """
    sentences = sent_tokenize(text)
    sentence_data = []

    for i, sent_text in enumerate(sentences):
        sent_text = sent_text.strip()

        # Skip empty, tiny, or legal citation sentences
        if len(sent_text) < 15 or is_legal_citation(sent_text):
            continue

        features = analyze_sentence(sent_text)

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
    Iterates through all supported files in a directory, extracts text,
    analyzes it for bias features, and saves the dataset to CSV.
    """
    print(f"Starting dataset creation from: {input_dir}")
    print(f"Output will be saved to: {output_csv}")

    all_sentences = []

    file_paths = list(set(
        glob.glob(os.path.join(input_dir, "*.PDF")) +
        glob.glob(os.path.join(input_dir, "*.pdf"))
    ))

    print(f"Found {len(file_paths)} PDF reports to process.")

    for fp in file_paths:
        filename = os.path.basename(fp)
        print(f"Processing: {filename}...")
        try:
            raw_text = extract_text(fp)
            doc_sentences = process_report(raw_text, filename)
            all_sentences.extend(doc_sentences)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(all_sentences)

    if df.empty:
        print("No valid sentences extracted!")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"\n--- SUCCESS ---")
    print(f"Extracted {len(df)} sentences across {len(file_paths)} reports.")
    print(f"Dataset saved successfully!")


if __name__ == "__main__":
    INPUT_DIR = r"c:\Users\Shambhavi\Desktop\forensic_dataset (1)"
    OUTPUT_FILE = r"c:\Users\Shambhavi\biasDetection\bias_dataset.csv"
    create_dataset(INPUT_DIR, OUTPUT_FILE)
