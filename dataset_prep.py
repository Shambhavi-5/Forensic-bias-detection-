import os, re, glob, string, pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from extract_text import extract_text

# --- Lexicons ---
ABSOLUTES = {"always", "never", "undoubtedly", "obviously", "certainly", "conclusively", "impossible", "clearly", "unquestionably", "irrefutable", "definite", "must", "proven", "proves", "unanimously", "absolute", "entirely", "completely", "certain", "sure", "conclusive", "indisputable", "evident", "plainly"}
PROCURAL_MARKERS = {"filed", "prescribed", "requirement", "compliance", "statutory", "administrative", "regulation", "provision", "procedure", "deadline", "certified", "mandated", "stipulated", "directed", "period", "days", "months", "section", "article", "rule", "code", "interpretation", "interpreted"}
SUBJECTIVES = {"shocking", "terrible", "bizarre", "suspicious", "horrific", "tragic", "disturbing", "unbelievable", "appalling", "ridiculous", "fortunately", "unfortunately", "sadly", "luckily", "surprisingly", "gruesome", "vicious", "cruel", "brutal", "heinous"}
HEDGES = {"might", "may", "could", "suggests", "appears", "seems", "possibly", "probably", "likely", "unlikely", "indicates", "implies", "perhaps", "tends", "assumes", "believes", "estimates", "maybe"}
OPINION_VERBS = {"believe", "think", "opine", "guess", "assume", "conclude", "seem", "seems", "appears", "appeared", "suggests", "suggested", "feel", "feels", "wonder", "estimate", "estimated", "speculate", "speculated"}
FACT_VERBS = {"found", "observed", "recorded", "noted", "seen", "conducted", "stated", "perused", "heard", "detailed", "documented", "recovered", "seized", "collected", "prepared", "submitted"}
NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "cannot", "doesn't", "don't", "does", "didn't", "wasn't", "isn't", "aren't", "couldn't", "wouldn't", "shouldn't"}

# --- Patterns ---
REPORTED_SPEECH_VERBS = re.compile(r'\b(argued|submitted|contended|alleged|claimed|stated|held|opined|observed|said|urged|insisted)\s+that\b', re.I)
CONDITIONAL_STARTERS = re.compile(r'^\s*(if|had|were|should|supposing|assuming|hypothetically|provided that|in the event)\b', re.I)

def is_legal_citation(text: str) -> bool:
    text = text.strip()
    if len(text) < 15 or 'indiankanoon.org' in text.lower() or re.match(r'^https?://', text, re.I): return True
    tokens = text.split()
    cit_count = sum(1 for tok in tokens if re.match(r'^[\d]+[.,/\-]?[\d]*$', tok) or 
                    re.match(r'^(s\.|sec\.|section|art\.|article|clause|order|rule|para|sub-section)', tok, re.I) or
                    re.match(r'^[A-Z]{2,}[-/]\d', tok) or re.match(r'^\(?[A-Z&]{2,5}\)?$', tok))
    return (cit_count / len(tokens)) > 0.5 if tokens else True

def strip_quoted_content(text: str) -> str:
    return re.sub(r'["\u201c\u201d][^"\u201c\u201d]*["\u201c\u201d]', ' ', text).strip()

def get_context_penalty(text: str) -> float:
    if CONDITIONAL_STARTERS.match(text): return 0.0
    return 0.5 if REPORTED_SPEECH_VERBS.search(text) else 1.0

def get_bias_types(abs_c, subj_c, hedge_c) -> list:
    res = []
    if abs_c > 0 and hedge_c == 0: res.append("Overconfidence Bias")
    if abs_c > hedge_c > 0: res.append("Anchoring Bias")
    if abs_c >= 2 and subj_c == 0 and hedge_c == 0 and "Overconfidence Bias" not in res: res.append("Confirmation Bias")
    return res if res else ["Unspecified Risk"]

def get_statement_nature(tokens, abs_c, subj_c, first_p_c) -> str:
    t_low = [t.lower() for t in tokens]
    if any(t in PROCURAL_MARKERS or t in {"heard", "perused", "submissions", "learned"} for t in t_low): return "Procedural"
    if any(t in OPINION_VERBS for t in t_low) or (first_p_c > 0 and not any(t in FACT_VERBS for t in t_low)) or (abs_c + subj_c) >= 2: return "Opinion"
    return "Fact"

def analyze_sentence(text: str) -> dict:
    penalty, clean_text = get_context_penalty(text), strip_quoted_content(text)
    tokens = word_tokenize(clean_text)
    tagged = pos_tag(tokens)
    abs_c, subj_c, hedge_c, first_p_c, word_c = 0, 0, 0, 0, 0
    matched_abs, matched_subj = [], []
    proc_ctx = any(m in clean_text.lower() for m in PROCURAL_MARKERS)

    for i, (tok, pos) in enumerate(tagged):
        if tok in string.punctuation: continue
        low, word_c = tok.lower(), word_c + 1
        prev = [tagged[j][0].lower() for j in range(max(0, i-2), i) if tagged[j][0] not in string.punctuation]
        negated = any(w in NEGATION_WORDS for w in prev)

        if low in ABSOLUTES and not negated:
            if (low == "must" and proc_ctx) or (low == "absolute" and any(x in clean_text.lower() for x in ["liability", "privilege"])): continue
            abs_c, _ = abs_c + 1, matched_abs.append(low)
        elif low in SUBJECTIVES and not negated:
            subj_c, _ = subj_c + 1, matched_subj.append(low)
        elif low in HEDGES: hedge_c += 1
        if pos in ("PRP", "PRP$") and low in {"i", "me", "my", "mine", "we", "us", "our"}: first_p_c += 1

    score = round((abs_c * 2) * penalty)
    b_types = get_bias_types(abs_c, subj_c, hedge_c)
    if penalty == 0: b_types = ["Conditional/Hypothetical Language"]
    elif penalty == 0.5 and score > 0: b_types = [b + " (Attributed)" for b in b_types]
    
    nature = get_statement_nature(tokens, abs_c, subj_c, first_p_c)
    return {"Word_Count": word_c, "Absolute_Count": abs_c, "Subjective_Count": subj_c, "Hedge_Count": hedge_c,
            "Hedge_Ratio": round(hedge_c/abs_c, 2) if abs_c > 0 else hedge_c, "First_Person_Count": first_p_c,
            "Bias_Risk_Score": score, "Bias_Types": ", ".join(b_types), "Key_Triggers": ", ".join(set(matched_abs + matched_subj)),
            "Is_Opinion": 1 if nature == "Opinion" else 0}

def is_def(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in [r"^the term .* means", r"^definition of .* is", r"^.* shall mean", r"^expressions? .* includes?"])

def get_contextual_metrics(win_feats: list) -> dict:
    scores = [f["Bias_Risk_Score"] for f in win_feats]
    return {"Contextual_Density": round(sum(scores)/len(scores), 2) if scores else 0, "Contextual_Momentum": scores[-1] - scores[0] if len(scores) > 1 else 0}

def process_report(text: str, doc_name: str) -> list:
    valid_sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= 15 and not is_legal_citation(s) and not is_def(s)]
    analyzed = [analyze_sentence(s) for s in valid_sents]
    res, win = [], 3
    for i in range(len(valid_sents)):
        ctx = get_contextual_metrics(analyzed[max(0, i-win//2) : min(len(valid_sents), i+win//2+1)])
        row = {"Document": doc_name, "Sentence_ID": i + 1, "Sentence_Text": valid_sents[i], "Context_Snippet": " | ".join(valid_sents[max(0, i-win//2) : min(len(valid_sents), i+win//2+1)])}
        row.update(analyzed[i]); row.update(ctx)
        res.append(row)
    return res

def create_dataset(in_dir, out_csv):
    files = list(set(glob.glob(os.path.join(in_dir, "*.PDF")) + glob.glob(os.path.join(in_dir, "*.pdf"))))
    data = []
    for fp in files:
        try: data.extend(process_report(extract_text(fp), os.path.basename(fp)))
        except Exception as e: print(f"Error {fp}: {e}")
    if not data: return print("No data!")
    if os.path.dirname(out_csv): os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    print(f"Saved {len(data)} rows to {out_csv}")

if __name__ == "__main__":
    create_dataset("forensic_dataset", "bias_dataset.csv")
