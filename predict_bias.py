import joblib
import pandas as pd
from dataset_prep import analyze_sentence # Import our NLTK feature extractor

# Map numeric predictions back to human-readable strings
RISK_MAPPING = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

def load_model(model_path: str):
    """Loads the pre-trained Random Forest model."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run model_training.py first?")

def predict_bias_risk(sentence: str, model) -> dict:
    """
    Takes a raw string sentence, extracts NLP features, and predicts
    its bias risk category using the trained model.
    """
    # 1. Extract linguistic features using the exact same logic used during training
    features = analyze_sentence(sentence)
    
    # The model was trained on ('Word_Count', 'Absolute_Count', 'Subjective_Count', 
    # 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count')
    # So we must provide a DataFrame matching that exact order
    
    feature_list = [
        features["Word_Count"],
        features["Absolute_Count"],
        features["Subjective_Count"],
        features["Hedge_Count"],
        features["Hedge_Ratio"],
        features["First_Person_Count"]
    ]
    
    columns = ['Word_Count', 'Absolute_Count', 'Subjective_Count', 
               'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count']
               
    X_new = pd.DataFrame([feature_list], columns=columns)
    
    # 2. Predict the risk tier
    prediction = model.predict(X_new)[0]
    
    # 3. Predict the probability (confidence)
    probabilities = model.predict_proba(X_new)[0]
    confidence = probabilities[prediction] * 100
    
    return {
        "Sentence": sentence,
        "Risk_Category": RISK_MAPPING[prediction],
        "Confidence": f"{confidence:.1f}%",
        "Extracted_Features": {
            "Absolutes": features["Absolute_Count"],
            "Subjectives": features["Subjective_Count"],
            "Hedges": features["Hedge_Count"]
        }
    }

if __name__ == "__main__":
    MODEL_PATH = r"c:\Users\Shambhavi\biasDetection\bias_model.pkl"
    model = load_model(MODEL_PATH)
    
    print("--- Bias Risk Inference Engine Ready ---\n")
    
    test_sentences = [
        # Should be Low Risk
        "The sample was collected from the desk at 9:00 AM.",
        
        # Should be Medium Risk (Hedge/Absolute mix)
        "It seems possible that the evidence always points to the suspect.",
        
        # Should be High Risk (Subjective/Absolute heavily penalized)
        "It is absolutely unquestionable that this horrific and brutal attack was committed by the accused."
    ]
    
    for sent in test_sentences:
        result = predict_bias_risk(sent, model)
        print(f"Text: '{result['Sentence']}'")
        print(f"Prediction: {result['Risk_Category']} (Confidence: {result['Confidence']})")
        print(f"Key Features Found: {result['Extracted_Features']}\n")
