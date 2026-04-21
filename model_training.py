import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def train_model(in_path, out_path):
    df = pd.read_csv(in_path)
    df['Risk'] = df['Bias_Risk_Score'].apply(lambda s: 0 if s == 0 else (1 if 1 <= s <= 2 else 2))
    feats = ['Word_Count', 'Evidence_Count', 'Reasoning_Count', 'Prosecution_Mentions', 'Defence_Mentions', 'Contradiction_Count', 'Certainty_Count', 'Subjective_Count', 'Certainty_Support_Ratio', 'Contextual_Imbalance']
    
    # Simple check to make sure stratify works if multiple classes aren't present
    classes = len(np.unique(df['Risk']))
    
    if classes < 2:
        print("Not enough classes in dataset to train a classification model. All data labeled as Risk 0.")
        return

    X_train, X_test, y_train, y_test = train_test_split(df[feats], df['Risk'], test_size=0.2, random_state=42, stratify=df['Risk'])
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    # Handle dynamic target names based on existing risk levels
    target_names = ["Low"]
    if classes >= 2: target_names.append("Med")
    if classes >= 3: target_names.append("High")
        
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n", classification_report(y_test, y_pred, target_names=target_names))
    
    joblib.dump(rf, out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    train_model("bias_dataset.csv", "bias_model.pkl")
