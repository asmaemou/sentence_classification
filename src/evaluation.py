"""
Only loads the cleaned test set,
Loads all models (including your stacked ensemble),
Transforms the test text with your saved TF-IDF vectorizer,
Evaluates each model (printing metrics, saving confusion matrices),
And at the end prints & saves a summary table.
"""
import os
import joblib
import pandas as pd
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

# --- Configuration -----------------------------------
POS_LABEL = "Positive"   # adjust if your CSV uses a different string
OUT_DIR = "./confusion_matrices"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data and vectorizer -----------------------
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")
test_df = test_df.sample(frac=0.1, random_state=42)   # smoke‐test subset
X_test = test_df["clean_review"].fillna("")
y_test = test_df["sentiment"]  # expected values: "Negative", "Positive"

vectorizer   = joblib.load("./models/tfidf_vectorizer.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# --- Helper to evaluate & save confusion matrix -----
def evaluate_model(model, X, y_true, name):
    y_pred = model.predict(X)
    
    # compute metrics in local vars (avoids multi-line f-string issues)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,
                           average='binary', pos_label=POS_LABEL)
    rec  = recall_score(y_true, y_pred,
                        average='binary', pos_label=POS_LABEL)
    f1   = f1_score(y_true, y_pred,
                    average='binary', pos_label=POS_LABEL)
    
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
    print(classification_report(
        y_true, y_pred,
        labels=["Negative","Positive"],
        target_names=["Negative","Positive"]
    ))
    
    # save confusion matrix
    cm = confusion_matrix(y_true, y_pred,
                          labels=["Negative","Positive"])
    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative","Positive"],
        yticklabels=["Negative","Positive"],
        cbar=False
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_cm.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")
    
    # return metrics for summary table
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

# --- Load and evaluate all models -------------------
models = {
    "Random Forest": joblib.load("./models/random_forest_model.pkl"),
    "Decision Tree": joblib.load("./models/decision_tree_model.pkl"),
    "XGBoost"      : joblib.load("./models/xgboost_model.pkl"),
    # "Stacked Model": joblib.load("./models/stacking_model.pkl")
}

rows = []
for name, mdl in models.items():
    metrics = evaluate_model(mdl, X_test_tfidf, y_test, name)
    rows.append(metrics)

# --- Summary table ----------------------------------
summary_df = pd.DataFrame(rows)
print("\n=== Summary of all models ===")
print(summary_df.to_string(index=False))

# optionally save summary to CSV
summary_df.to_csv("model_metrics_summary.csv", index=False)
print("Saved summary table to model_metrics_summary.csv")
