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
    "unanimously", "absolute", "entirely", "completely",
    "certain", "sure", "conclusive", "indisputable", "evident", "plainly"
}

# 2. Indicators of Procedural/Statutory Language (used to neutralize false positives)
PROCURAL_MARKERS = {
    "filed", "prescribed", "requirement", "compliance", "statutory", "administrative",
    "regulation", "provision", "procedure", "deadline", "certified", "mandated",
    "stipulated", "directed", "period", "days", "months", "section", "article",
    "rule", "code", "interpretation", "interpreted"
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

# 3. Opinion-signaling verbs (Expert Opinion / Subjectivity)
OPINION_VERBS = {
    "believe", "think", "opine", "guess", "assume", "conclude", "seem", 
    "seems", "appears", "appeared", "suggests", "suggested", "feel", 
    "feels", "wonder", "estimate", "estimated", "speculate", "speculated"
}

# 4. Fact-signaling verbs (Objective Reporting)
FACT_VERBS = {
    "found", "found", "observed", "recorded", "noted", "seen", "noted", 
    "conducted", "stated", "perused", "heard", "detailed", "documented",
    "recovered", "seized", "collected", "prepared", "submitted"
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


def get_statement_nature(tokens: list, tagged_tokens: list, absolute_count: int, 
                         subjective_count: int, first_person_count: int) -> str:
    """
    Classifies the sentence as 'Opinion', 'Fact', or 'Procedural'.
    Logic:
    - Opinion: High 1st person + opinion verbs OR high subjective word density.
    - Procedural: Presence of PROCURAL_MARKERS or specific formulaic phrases.
    - Fact: Dominated by Fact verbs, past tense, and low subjectivity.
    """
    text_lower = " ".join(tokens).lower()
    
    has_opinion_verb = any(t in OPINION_VERBS for t in tokens)
    has_fact_verb = any(t in FACT_VERBS for t in tokens)
    
    # 1. Procedural Check (highest priority)
    procedural_terms = {"heard", "perused", "submissions", "counsel", "learned", "case file"}
    if any(term in text_lower for term in procedural_terms) or \
       any(m in text_lower for m in PROCURAL_MARKERS):
        return "Procedural"

    # 2. Opinion Check
    # "In my opinion", "I believe", "it appears"
    if has_opinion_verb or (first_person_count > 0 and not has_fact_verb):
        return "Opinion"
    
    # High adjective/adverb density often indicates opinion framing
    if (absolute_count + subjective_count) >= 2:
        return "Opinion"

    # 3. Fact Check
    if has_fact_verb:
        return "Fact"

    # Default to Fact for neutral technical reporting
    return "Fact"


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

        # Step 4: Procedural Neutralizer
        # If absolute words like 'must' or 'absolute' appear near legal markers (Section, Rule, filed),
        # they are likely procedural/statutory rather than subjective overconfidence.
        is_procedural_context = any(m in analysis_text.lower() for m in PROCURAL_MARKERS)
        
        # Step 5: Negation check — look back 1-2 tokens for negation words
        preceding_words = [
            tagged_tokens[i][0].lower()
            for i in range(max(0, idx - 2), idx)
            if tagged_tokens[i][0] not in string.punctuation
        ]
        is_negated = any(w in NEGATION_WORDS for w in preceding_words)

        # Check against lexicons
        # Neutralizer: 'must' is only a bias if it's not in a procedural context
        if word_lower in ABSOLUTES and not is_negated:
            if word_lower == "must" and is_procedural_context:
                continue # Skip neutral procedural 'must'
            if word_lower == "absolute" and ("liability" in analysis_text.lower() or "privilege" in analysis_text.lower()):
                continue # Skip legal terms of art
                
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

    # Step 8: Differentiate Fact vs. Opinion
    statement_nature = get_statement_nature(tokens, tagged_tokens, absolute_count, 
                                          subjective_count, first_person_count)
    is_opinion = 1 if statement_nature == "Opinion" else 0

    return {
        "Word_Count": word_count,
        "Absolute_Count": absolute_count,
        "Subjective_Count": subjective_count,
        "Hedge_Count": hedge_count,
        "Hedge_Ratio": round(hedge_ratio, 2),
        "First_Person_Count": first_person_count,
        "Bias_Risk_Score": bias_score,
        "Bias_Types": ", ".join(bias_types),
        "Key_Triggers": ", ".join(list(set(matched_absolutes + matched_subjectives))),
        "Statement_Nature": statement_nature,
        "Is_Opinion": is_opinion
    }

def is_definition_sentence(text: str) -> bool:
    """Detects if a sentence is just a definition, which shouldn't be flagged as bias."""
    low_text = text.lower()
    patterns = [
        r"^the term .* means",
        r"^definition of .* is",
        r"^.* shall mean",
        r"^expressions? .* includes?"
    ]
    return any(re.search(p, low_text) for p in patterns)


def get_contextual_metrics(window_features: list) -> dict:
    """
    Calculates metrics across a window of analyzed sentences.
    - Contextual_Density: Avg bias score in the window.
    - Contextual_Momentum: Difference in scores (detects escalating bias).
    """
    scores = [f["Bias_Risk_Score"] for f in window_features]
    
    # Average score in the local neighborhood
    density = sum(scores) / len(scores) if scores else 0
    
    # Momentum: is the bias increasing? (last - first)
    momentum = scores[-1] - scores[0] if len(scores) > 1 else 0
    
    return {
        "Contextual_Density": round(density, 2),
        "Contextual_Momentum": momentum
    }


# =============================================================================
# DATASET GENERATION
# =============================================================================

def process_report(text: str, document_name: str) -> list:
    """
    Processes the full text of one report using a sliding window (size=3)
    to provide contextual awareness.
    """
    sentences = sent_tokenize(text)
    
    # Filter out empty/citations first to maintain sequence integrity
    valid_sentences = []
    for i, sent_text in enumerate(sentences):
        sent_text = sent_text.strip()

        # Skip empty, tiny, legal citations, or definition sentences
        if len(sent_text) < 15 or is_legal_citation(sent_text) or is_definition_sentence(sent_text):
            continue
        valid_sentences.append(sent_text)
            
    # Analyze all sentences first to gather features
    analyzed_list = [analyze_sentence(s) for s in valid_sentences]
    
    sentence_data = []
    window_size = 3
    
    for i in range(len(valid_sentences)):
        # Define window (previous, current, next)
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(valid_sentences), i + window_size // 2 + 1)
        window_features = analyzed_list[start_idx:end_idx]
        
        # Calculate contextual metrics
        context_metrics = get_contextual_metrics(window_features)
        
        # Build combined row
        current_features = analyzed_list[i]
        row = {
            "Document": document_name,
            "Sentence_ID": i + 1,
            "Sentence_Text": valid_sentences[i],
            "Context_Snippet": " | ".join(valid_sentences[start_idx:end_idx])
        }
        row.update(current_features)
        row.update(context_metrics)
        
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
