"""
Only loads the cleaned test set


Loads all four models (including your stacked ensemble).

Transforms the test text with your saved TF-IDF vectorizer.

Evaluates each model and saves confusion matrices.
"""
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Create output directory
os.makedirs("./confusion_matrices", exist_ok=True)

# 1) Load only the cleaned test set
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")

# — for a quick smoke-test on 10% — 
test_df = test_df.sample(frac=0.1, random_state=42)

# 2) Extract features and labels
X_test = test_df["clean_review"].fillna("")
y_test = test_df["sentiment"]

# 3) Load TF-IDF vectorizer and transform
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# 4) Helper to evaluate & save confusion matrix
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y, y_pred):.4f}\n")
    print(classification_report(y, y_pred))
    
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"./confusion_matrices/{name.replace(' ', '_')}_cm.png")
    plt.close()

# 5) Load all models (including stacking)
models = {
    "Random Forest": joblib.load("./models/random_forest_model.pkl"),
    "Decision Tree": joblib.load("./models/decision_tree_model.pkl"),
    "XGBoost"      : joblib.load("./models/xgboost_model.pkl"),
    "Stacked Model": joblib.load("./models/stacking_model.pkl")  # ensure you saved this in modeling.py
}

# 6) Evaluate each
for name, mdl in models.items():
    evaluate_model(mdl, X_test_tfidf, y_test, name)

print("\nEvaluation completed successfully!")
