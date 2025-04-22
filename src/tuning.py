import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint, uniform

# Create the models directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Load the preprocessed dataset
train_df = pd.read_csv("../data/Amazon_reviews/train_cleaned_final.csv")
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")

# Extract features and labels
X_train = train_df["clean_review"]
y_train = train_df["sentiment"]
X_test = test_df["clean_review"]
y_test = test_df["sentiment"]

# Handle missing values by filling NaNs with empty strings
X_train.fillna("", inplace=True)
X_test.fillna("", inplace=True)

# ------------------------------------------------------------------------------
# Vectorizing the text (TF-IDF)
# Increased max_features and added ngram_range to capture more contextual information
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the vectorizer
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

# ------------------------------------------------------------------------------
# Split the training data further into training and validation sets (used for early stopping in XGBoost)
X_train_subset, X_val_subset, y_train_subset, y_val_subset = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ------------------------------------------------------------------------------
# Randomized Search for Random Forest
rf_param_dist = {
    'n_estimators': randint(100, 300),       # Expanded estimator range for potential performance gain
    'max_depth': [10, 20, None],               # Allowing None to let trees grow if needed
    'min_samples_split': randint(2, 10),       # A bit wider range for split values
    'min_samples_leaf': randint(1, 10),        # A bit wider range for leaf sizes
    'bootstrap': [True, False]               # Allow both to see if not bootstrapping improves performance
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_dist,
    n_iter=50,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    random_state=42
)
rf_random_search.fit(X_train_subset, y_train_subset)

print(f"Best Parameters for Random Forest: {rf_random_search.best_params_}")
print(f"Best Score for Random Forest: {rf_random_search.best_score_:.4f}")

# ------------------------------------------------------------------------------
# Randomized Search for Decision Tree
dt_param_dist = {
    'max_depth': [10, 20, None],               # Including None to grow fully if needed
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']          # Testing both criteria
}

dt_model = DecisionTreeClassifier(random_state=42)
dt_random_search = RandomizedSearchCV(
    estimator=dt_model,
    param_distributions=dt_param_dist,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    random_state=42
)
dt_random_search.fit(X_train_subset, y_train_subset)

print(f"Best Parameters for Decision Tree: {dt_random_search.best_params_}")
print(f"Best Score for Decision Tree: {dt_random_search.best_score_:.4f}")

# ------------------------------------------------------------------------------
# Randomized Search for XGBoost
# Note: Removed "use_label_encoder" as it is deprecated and unnecessary.
xgb_param_dist = {
    'learning_rate': uniform(0.01, 0.1),     # Wider range to experiment with slower and faster learning rates
    'n_estimators': randint(100, 300),         # Expanded range for number of trees
    'max_depth': randint(3, 8),                # Adjusted depth range to allow more complex trees if needed
    'subsample': uniform(0.7, 0.3),            # Testing smaller subsample values
    'colsample_bytree': uniform(0.7, 0.3)       # Testing smaller colsample values as well
}

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)

# Prepare fit parameters to add early stopping using the validation set.
fit_params = {
    "eval_set": [(X_val_subset, y_val_subset)],
    "early_stopping_rounds": 10,
    "verbose": False
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    random_state=42
)
xgb_random_search.fit(X_train_subset, y_train_subset, **fit_params)

print(f"Best Parameters for XGBoost: {xgb_random_search.best_params_}")
print(f"Best Score for XGBoost: {xgb_random_search.best_score_:.4f}")

# ------------------------------------------------------------------------------
# Save the best models from tuning
joblib.dump(rf_random_search.best_estimator_, "./models/random_forest_best_model.pkl")
joblib.dump(dt_random_search.best_estimator_, "./models/decision_tree_best_model.pkl")
joblib.dump(xgb_random_search.best_estimator_, "./models/xgboost_best_model.pkl")

# ------------------------------------------------------------------------------
# Function for evaluating models on the test set
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name} Model:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Load the best models from tuning
rf_best_model = joblib.load("./models/random_forest_best_model.pkl")
dt_best_model = joblib.load("./models/decision_tree_best_model.pkl")
xgb_best_model = joblib.load("./models/xgboost_best_model.pkl")

# Evaluate the models
evaluate_model(rf_best_model, X_test_tfidf, y_test, "Random Forest")
evaluate_model(dt_best_model, X_test_tfidf, y_test, "Decision Tree")
evaluate_model(xgb_best_model, X_test_tfidf, y_test, "XGBoost")

print("\nHyperparameter tuning and evaluation completed successfully!")

