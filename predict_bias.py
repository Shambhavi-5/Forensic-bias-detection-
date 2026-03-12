import joblib, pandas as pd
from dataset_prep import analyze_sentence

RISK_MAPPING = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
FEAT_COLS = ['Word_Count', 'Absolute_Count', 'Subjective_Count', 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count', 'Contextual_Density', 'Contextual_Momentum', 'Is_Opinion']

def load_model(path):
    try: return joblib.load(path)
    except FileNotFoundError: raise FileNotFoundError(f"Model not found at {path}. Run model_training.py first.")

def predict_bias_risk(sentence, model, density=0.0, momentum=0.0):
    f = analyze_sentence(sentence)
    vals = [f[c] if c in f else (density if c == 'Contextual_Density' else momentum) for c in FEAT_COLS]
    X = pd.DataFrame([vals], columns=FEAT_COLS)
    pred = model.predict(X)[0]
    return {"Sentence": sentence, "Risk_Category": RISK_MAPPING[pred], "Confidence": f"{model.predict_proba(X)[0][pred]*100:.1f}%",
            "Features": {k: f.get(k, 0) for k in ["Absolute_Count", "Subjective_Count", "Hedge_Count", "Is_Opinion"]}}

if __name__ == "__main__":
    model = load_model("bias_model.pkl")
    for s in ["The sample was collected at 9 AM.", "It seems possible that the evidence always points to the suspect.", "It is absolutely unquestionable that this horrific attack was committed by the accused."]:
        r = predict_bias_risk(s, model)
        print(f"Text: '{r['Sentence']}'\nPrediction: {r['Risk_Category']} ({r['Confidence']})\nFeatures: {r['Features']}\n")
