import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def create_risk_tiers(score):
    """
    Converts the continuous Bias_Risk_Score into discrete categories:
    0: Low Risk (Score == 0)
    1: Medium Risk (Score between 1 and 3)
    2: High Risk (Score > 3)
    """
    if score == 0:
        return 0
    elif 1 <= score <= 3:
        return 1
    else:
        return 2

def train_model(dataset_path: str, model_output_path: str):
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 1. Prepare Target Variable (Y)
    df['Risk_Category'] = df['Bias_Risk_Score'].apply(create_risk_tiers)
    
    print("\nClass Distribution:")
    print(df['Risk_Category'].value_counts().sort_index().rename({0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}))
    
    # 2. Select Features (X)
    # We drop identifying/text columns and the original score we are trying to predict
    features = ['Word_Count', 'Absolute_Count', 'Subjective_Count', 
                'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count']
                
    X = df[features]
    y = df['Risk_Category']
    
    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)} samples...")
    
    # 4. Initialize and Train the Random Forest Model
    # We use class_weight='balanced' because 'High Risk' cases are usually a minority
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    y_pred = rf_model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    
    target_names = ['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Print Feature Importance to see what the model cares about most
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance ---")
    print(feature_importance_df.to_string(index=False))
    
    # 6. Save the Model
    joblib.dump(rf_model, model_output_path)
    print(f"\n[+] Model successfully saved to: {model_output_path}")

if __name__ == "__main__":
    DATASET_PATH = r"c:\Users\Shambhavi\biasDetection\bias_dataset.csv"
    MODEL_PATH = r"c:\Users\Shambhavi\biasDetection\bias_model.pkl"
    train_model(DATASET_PATH, MODEL_PATH)
