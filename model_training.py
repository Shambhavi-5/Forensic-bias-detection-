import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(in_path, out_path):
    df = pd.read_csv(in_path)
    df['Risk'] = df['Bias_Risk_Score'].apply(lambda s: 0 if s == 0 else (1 if 1 <= s <= 3 else 2))
    feats = ['Word_Count', 'Absolute_Count', 'Subjective_Count', 'Hedge_Count', 'Hedge_Ratio', 'First_Person_Count', 'Contextual_Density', 'Contextual_Momentum', 'Is_Opinion']
    
    X_train, X_test, y_train, y_test = train_test_split(df[feats], df['Risk'], test_size=0.2, random_state=42, stratify=df['Risk'])
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n", classification_report(y_test, y_pred, target_names=['Low', 'Med', 'High']))
    
    joblib.dump(rf, out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    train_model("bias_dataset.csv", "bias_model.pkl")
