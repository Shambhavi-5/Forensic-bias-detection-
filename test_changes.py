from dataset_prep import analyze_sentence, is_legal_citation

citations = [
    "I have heard learned counsel and perused the case file in detail.",
    "Indian Kanoon - http://indiankanoon.org/doc/183744109/",
    "Section 168 of the Motor Vehicles Act, 1988.",
    "FAO-6160-2023 (O&M) -6- Page 3 of 36"
]

prose = [
    "It is absolutely unquestionable that this horrific and brutal attack was committed by the accused.",
    "The accused must have clearly known the consequences of his actions.",
    "In my opinion, there cannot be any hard and fast rule."
]

print("--- Citation Filter Test ---")
for s in citations:
    result = is_legal_citation(s)
    print("Skip=" + str(result) + ": " + s[:70])

print()
print("--- Bias Type Detection Test ---")
for s in prose:
    feat = analyze_sentence(s)
    score = feat["Bias_Risk_Score"]
    bias_types = feat["Bias_Types"]
    print("Score=" + str(score) + " Types=[" + bias_types + "]")
    print("  >> " + s[:80])
