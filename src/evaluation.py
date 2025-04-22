import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving confusion matrices
os.makedirs("./confusion_matrices", exist_ok=True)

# Load the preprocessed dataset
train_df = pd.read_csv("../data/Amazon_reviews/train_cleaned_final.csv")
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")

# Extract features and labels
X_train = train_df["clean_review"]
y_train = train_df["sentiment"]
X_test = test_df["clean_review"]
y_test = test_df["sentiment"]

# Handle missing values in the 'clean_review' column (fill NaNs with empty strings)
X_train.fillna("", inplace=True)
X_test.fillna("", inplace=True)

# Load the TF-IDF vectorizer 
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

# Transform the text data into TF-IDF features
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to evaluate the model and save confusion matrix
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name} Model:")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Display the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    
    # Save the confusion matrix plot
    cm_path = os.path.join("./confusion_matrices", f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()  # Close the plot to free memory

# Load the best models from tuning
rf_best_model = joblib.load("./models/random_forest_model.pkl")
dt_best_model = joblib.load("./models/decision_tree_model.pkl")
xgb_best_model = joblib.load("./models/xgboost_model.pkl")

# Evaluate the models
evaluate_model(rf_best_model, X_test_tfidf, y_test, "Random Forest")
evaluate_model(dt_best_model, X_test_tfidf, y_test, "Decision Tree")
evaluate_model(xgb_best_model, X_test_tfidf, y_test, "XGBoost")

print("\n  Evaluation completed successfully!")