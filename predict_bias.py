import joblib, pandas as pd
from dataset_prep import analyze_sentence

RISK_MAPPING = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
FEAT_COLS = ['Word_Count', 'Evidence_Count', 'Reasoning_Count', 'Prosecution_Mentions', 'Defence_Mentions', 'Contradiction_Count', 'Certainty_Count', 'Subjective_Count', 'Certainty_Support_Ratio', 'Contextual_Imbalance']

def load_model(path):
    try: return joblib.load(path)
    except FileNotFoundError: raise FileNotFoundError(f"Model not found at {path}. Run model_training.py first.")

def predict_bias_risk(sentence, model, imbalance=0.0):
    f = analyze_sentence(sentence)
    vals = [f[c] if c in f else imbalance for c in FEAT_COLS]
    X = pd.DataFrame([vals], columns=FEAT_COLS)
    pred = model.predict(X)[0]
    return {
        "Sentence": sentence, 
        "Risk_Category": RISK_MAPPING[pred], 
        "Confidence": f"{model.predict_proba(X)[0][pred]*100:.1f}%",
        "Features": {k: f.get(k, 0) for k in ["Evidence_Count", "Reasoning_Count", "Certainty_Support_Ratio"]}
    }

if __name__ == "__main__":
    model = load_model("bias_model.pkl")
    for s in ["The evidence clearly points to the suspect.", "There is no material before the Court to prove this point.", "It is absolutely unquestionable without a doubt."]:
        r = predict_bias_risk(s, model)
        print(f"Text: '{r['Sentence']}'\nPrediction: {r['Risk_Category']} ({r['Confidence']})\nFeatures: {r['Features']}\n")