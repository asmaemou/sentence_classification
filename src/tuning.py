import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

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

# Vectorizing the text (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the vectorizer
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

# Split the training data further into training and validation sets
X_train_subset, X_val_subset, y_train_subset, y_val_subset = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Hyperparameter tuning for Random Forest using GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
rf_grid_search.fit(X_train_subset, y_train_subset)

print(f"Best Parameters for Random Forest: {rf_grid_search.best_params_}")
print(f"Best Score for Random Forest: {rf_grid_search.best_score_}")

# Hyperparameter tuning for Decision Tree using GridSearchCV
dt_param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_model = DecisionTreeClassifier(random_state=42)

dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
dt_grid_search.fit(X_train_subset, y_train_subset)

print(f"Best Parameters for Decision Tree: {dt_grid_search.best_params_}")
print(f"Best Score for Decision Tree: {dt_grid_search.best_score_}")

# Hyperparameter tuning for XGBoost using GridSearchCV
xgb_param_grid = {
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
xgb_grid_search.fit(X_train_subset, y_train_subset)

print(f"Best Parameters for XGBoost: {xgb_grid_search.best_params_}")
print(f"Best Score for XGBoost: {xgb_grid_search.best_score_}")

# Save the best models from tuning
joblib.dump(rf_grid_search.best_estimator_, "./models/random_forest_best_model.pkl")
joblib.dump(dt_grid_search.best_estimator_, "./models/decision_tree_best_model.pkl")
joblib.dump(xgb_grid_search.best_estimator_, "./models/xgboost_best_model.pkl")

# Evaluate models on the test set
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

print("\n Hyperparameter tuning and evaluation completed successfully!")
