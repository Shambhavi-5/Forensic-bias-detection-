"""
Diagnostic script to research Fact vs. Opinion features.
Testing linguistic triggers for subjective statements vs. objective reporting.
"""
import nltk
from nltk import pos_tag, word_tokenize

# Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

test_cases = {
    "Direct Fact (Objective)": [
        "The weapon was found in the bedroom cabinet.",
        "The post-mortem was conducted by Dr. Singh on July 4th.",
        "The accused was seen entering the house at 10 PM.",
        "The samples were collected and sealed for testing."
    ],
    "First-Person Procedural (Objective)": [
        "I have examined the case file in detail.",
        "I visited the scene of crime on Monday.",
        "I found no merit in these submissions."
    ],
    "Expert Opinion (Subjective)": [
        "In my opinion, the injury seems self-inflicted.",
        "I believe the accused had a clear motive.",
        "I am certain that the timeline is impossible."
    ],
    "Third-Person Bias (Subjective)": [
        "The evidence conclusively proves the horrific nature of the crime.",
        "It is obviously a case of premeditated murder.",
        "Clearly, the prosecution story is Believable."
    ]
}

def check_fact_opinion_signals(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    # Subjective indicators
    adj_count = len([t for t, p in tagged if p.startswith('JJ')])
    adv_count = len([t for t, p in tagged if p.startswith('RB')])
    
    # First person
    first_person = len([t for t in tokens if t.lower() in {"i", "me", "my", "mine", "we", "us", "our"}])
    
    # Opinion verbs
    opinion_verbs = {"believe", "think", "opine", "seems", "appears", "feel", "suggests", "conclude", "certain", "uncertain"}
    has_opinion_verb = any(t.lower() in opinion_verbs for t in tokens)
    
    # Fact verbs
    fact_verbs = {"found", "conducted", "seen", "collected", "examined", "visited", "stated", "perused", "heard"}
    has_fact_verb = any(t.lower() in fact_verbs for t in tokens)

    return {
        "Adj/Adv": adj_count + adv_count,
        "1stP": first_person,
        "OpV": has_opinion_verb,
        "FaV": has_fact_verb
    }

print(f"{'Category/Sentence':<60} | {'A/A':<3} | {'1st':<3} | {'OpV':<3} | {'FaV':<3}")
print("-" * 80)

for cat, sentences in test_cases.items():
    print(f"\n[{cat}]")
    for sent in sentences:
        signals = check_fact_opinion_signals(sent)
        print(f"{sent[:58]:<60} | {signals['Adj/Adv']:<3} | {signals['1stP']:<3} | {str(signals['OpV'])[0]:<3} | {str(signals['FaV'])[0]:<3}")
