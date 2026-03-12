"""
Diagnostic script to verify false positives in bias detection.
Tests sentences that contain absolute words like 'must' in a factual or legal context.
"""
from dataset_prep import analyze_sentence

test_cases = [
    "The application must be filed within 30 days.",
    "Section 168 provides that the amount of compensation must be fair.",
    "The term 'just' must be interpreted according to legal standards.",
    "The prosecution must prove its case beyond reasonable doubt.",
    "This court must ensure a fair trial for the accused.",
    "The accused must have planned the attack in advance.", # Likely bias
    "It is absolute liability in such industrial accidents.",
    "The record must be certified as per Section 65B.",
    "I am certain that the accused is guilty.", # Bias
    "The court is certainly satisfied with the evidence.", # Bias
]

print(f"{'Sentence':<60} | {'Score':<5} | {'Types'}")
print("-" * 85)

for sent in test_cases:
    res = analyze_sentence(sent)
    print(f"{sent[:60]:<60} | {res['Bias_Risk_Score']:<5} | {res['Bias_Types']}")
