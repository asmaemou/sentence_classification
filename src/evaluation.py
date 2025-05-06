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
    confusion_matrix,
    classification_report
)

# --- Configuration ---
OUT_DIR = "./confusion_matrices"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load Test Data & Vectorizer ---
test_df = (
    pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")
      .sample(frac=0.1, random_state=42)    
)
X_test = test_df["clean_review"].fillna("")
y_test = test_df["sentiment"]              

vectorizer   = joblib.load("./models/tfidf_vectorizer.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# --- Evaluation Helper ---
def evaluate_model(model, X, y_true, name):

    y_pred = model.predict(X)

    # --- compute macro-averaged metrics  ---
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")

    # --- print ---
    print(f"\n=== {name} ===")
    print(f"Accuracy (overall)     : {acc:.4f}")
    print(f"Precision (macro avg.) : {prec:.4f}")
    print(f"Recall    (macro avg.) : {rec:.4f}")
    print(f"F1-Score  (macro avg.) : {f1:.4f}\n")
    print(classification_report(y_true, y_pred))

    # --- plot & save confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # --- return metrics for summary table ---
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

# --- Load Models & Run Evaluation ---
models = {
    "Random Forest": joblib.load("./models/random_forest_model.pkl"),
    "Decision Tree": joblib.load("./models/decision_tree_model.pkl"),
    "XGBoost"      : joblib.load("./models/xgboost_model.pkl"),

}

rows = []
for name, mdl in models.items():
    metrics = evaluate_model(mdl, X_test_tfidf, y_test, name)
    rows.append(metrics)

# --- Build & Save Summary Table ---
summary_df = pd.DataFrame(rows).set_index("Model")
print("\n=== Summary of all models ===")
print(summary_df)

summary_csv = "model_metrics_summary.csv"
summary_df.to_csv(summary_csv)
print(f"Saved summary table to {summary_csv}")
