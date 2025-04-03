import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Create the models directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Load the preprocessed dataset
train_df = pd.read_csv("../data/Amazon_reviews/train_cleaned.csv")
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned.csv")

# Extract features and labels
X_train = train_df["clean_review"]
y_train = train_df["sentiment"]
X_test = test_df["clean_review"]
y_test = test_df["sentiment"]

# Optionally, sample a smaller subset (e.g., 10% of the data)
train_df_subset = train_df.sample(frac=0.1, random_state=42)
test_df_subset = test_df.sample(frac=0.1, random_state=42)

X_train = train_df_subset["clean_review"]
y_train = train_df_subset["sentiment"]
X_test = test_df_subset["clean_review"]
y_test = test_df_subset["sentiment"]

# Handle missing values by filling NaNs with empty strings
X_train.fillna("", inplace=True)
X_test.fillna("", inplace=True)

# Vectorizing the text (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the vectorizer
print("Saving tfidf vectorizer ")
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

# Split the training data further into training and validation sets
X_train_subset, X_val_subset, y_train_subset, y_val_subset = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Random Forest Classifier
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)
    return model

# Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# XGBoost Classifier
def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    return model

# Train models
rf_model = train_random_forest(X_train_subset, y_train_subset)
dt_model = train_decision_tree(X_train_subset, y_train_subset)
xgb_model = train_xgboost(X_train_subset, y_train_subset)

# Save models
print("Saving Random Forest Model...")
joblib.dump(rf_model, "./models/random_forest_model.pkl")
print("Random Forest Model Saved!")

print("Saving Decision Tree Model...")
joblib.dump(dt_model, "./models/decision_tree_model.pkl")
print("Decision Tree Model Saved!")

print("Saving XGBoost Model...")
joblib.dump(xgb_model, "./models/xgboost_model.pkl")
print("XGBoost Model Saved!")

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name} Model:")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# Evaluate Random Forest on the validation set
evaluate_model(rf_model, X_val_subset, y_val_subset, "Random Forest")

# Evaluate Decision Tree on the validation set
evaluate_model(dt_model, X_val_subset, y_val_subset, "Decision Tree")

# Evaluate XGBoost on the validation set
evaluate_model(xgb_model, X_val_subset, y_val_subset, "XGBoost")

# Evaluate models on the test set
evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest (Test Set)")
evaluate_model(dt_model, X_test_tfidf, y_test, "Decision Tree (Test Set)")
evaluate_model(xgb_model, X_test_tfidf, y_test, "XGBoost (Test Set)")

print("\n Model training and evaluation completed successfully!")
