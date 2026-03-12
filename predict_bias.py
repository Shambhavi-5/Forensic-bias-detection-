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

def predict_bias_risk(sentence: str, model, context_density: float = 0.0, context_momentum: float = 0.0) -> dict:
    """
    Takes a raw string sentence, extracts isolation features, and combined with 
    provided context, predicts the bias risk category.
    """
    # 1. Extract linguistic features in isolation
    features = analyze_sentence(sentence)
    
    # Combined feature list for the model (9 features)
    feature_list = [
        features["Word_Count"],
        features["Absolute_Count"],
        features["Subjective_Count"],
        features["Hedge_Count"],
        features["Hedge_Ratio"],
        features["First_Person_Count"],
        context_density,
        context_momentum,
        features["Is_Opinion"]
    ]
    
    columns = [
        'Word_Count', 'Absolute_Count', 'Subjective_Count', 
        'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count',
        'Contextual_Density', 'Contextual_Momentum', 'Is_Opinion'
    ]
               
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
        "Nature": features["Statement_Nature"],
        "Extracted_Features": {
            "Absolutes": features["Absolute_Count"],
            "Subjectives": features["Subjective_Count"],
            "Hedges": features["Hedge_Count"],
            "Density": context_density,
            "Is_Opinion": features["Is_Opinion"]
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
