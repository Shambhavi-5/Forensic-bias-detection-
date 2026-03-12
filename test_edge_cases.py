"""
Edge Case Analysis Script for Bias Detection Pipeline.
Tests each candidate edge case to understand how current detection handles it.
"""
from dataset_prep import analyze_sentence, is_legal_citation

test_cases = {
    "1. Quoted court holding (author quotes someone else's words)": [
        'has held that: "Motor Vehicles Act, Section 168 provides that amount of compensation appears to it to be just."',
        'The court stated: "Obviously just compensation does not mean perfect or absolute compensation."',
        '"The accused is clearly guilty," the complainant alleged.',
    ],
    "2. Negated absolutes (should NOT be bias - word is being denied)": [
        "The evidence does not conclusively prove the accused was present.",
        "There is absolutely no basis for this claim.",
        "This cannot never be proven beyond doubt.",
    ],
    "3. Reported speech / attribution": [
        "The prosecution argued that the accused must have been at the scene.",
        "The defence counsel submitted that clearly no motive existed.",
        "The complainant alleged that the accused obviously planned this.",
    ],
    "4. Legal boilerplate formulaic phrases": [
        "I find no merit in the submissions advanced by learned counsel for the appellants.",
        "This court has heard all parties and perused the record carefully.",
        "The appeal is accordingly dismissed. No order as to costs.",
    ],
    "5. Technical statutory definitions (not author's opinion)": [
        "The term 'just' means that the amount so determined is fair and reasonable.",
        "Section 65B provides that electronic records must be certified.",
    ],
    "6. Conditional or hypothetical language": [
        "If the accused had clearly intended harm, the charge would have been murder.",
        "Had this evidence been proven conclusively, the verdict would differ.",
    ],
    "7. Standard conviction/acquittal phrases": [
        "The accused is undoubtedly convicted under Section 302 IPC.",
        "This court is clearly satisfied that the prosecution has proven its case.",
    ],
}

print("=" * 80)
print("EDGE CASE ANALYSIS")
print("=" * 80)

for category, sentences in test_cases.items():
    print(f"\n### {category}")
    for sent in sentences:
        features = analyze_sentence(sent)
        skip = is_legal_citation(sent)
        print(f"  Skip={skip} | Score={features['Bias_Risk_Score']} | Types=[{features['Bias_Types']}]")
        print(f"  >> {sent[:90]}")
