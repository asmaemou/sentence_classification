import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint, uniform
import joblib
import os

# ——— CONFIGURATION —————————————
SUBSAMPLE_FRAC = 0.1    
RF_ITERS     = 10       
DT_ITERS     = 5
XGB_ITERS    = 5

# 1) Prepare directories
os.makedirs("./models", exist_ok=True)

# 2) Load cleaned data
train_df = pd.read_csv("../data/Amazon_reviews/train_cleaned_final.csv")
test_df  = pd.read_csv("../data/Amazon_reviews/test_cleaned_final.csv")
# Remap sentiments from {1,2} → {0,1}
train_df['sentiment'] = train_df['sentiment'] - 1
test_df['sentiment']  = test_df['sentiment']  - 1
# 3) Subsample for a quick run
train_df = train_df.sample(frac=SUBSAMPLE_FRAC, random_state=42)
test_df  = test_df.sample(frac=SUBSAMPLE_FRAC, random_state=42)

# 4) Extract features & labels
X_train = train_df["clean_review"].fillna("")
y_train = train_df["sentiment"]
X_test  = test_df["clean_review"].fillna("")
y_test  = test_df["sentiment"]

# 5) TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,     
    stop_words='english',
    ngram_range=(1,2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 6) Save vectorizer
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

# 7) Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tfidf, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# 8) Set up hyperparameter grids 
rf_dist = {
    'n_estimators': randint(50,150),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}
dt_dist = {
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 5),
    'criterion': ['gini', 'entropy']
}
xgb_dist = {
    'learning_rate': uniform(0.01, 0.1),
    'n_estimators': randint(50,150),
    'max_depth': randint(3,6),
    'subsample': uniform(0.7,0.3),
    'colsample_bytree': uniform(0.7,0.3)
}

# 9) Randomized searches
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_dist, n_iter=RF_ITERS, cv=2, n_jobs=-1, verbose=1, random_state=42
)
dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_dist, n_iter=DT_ITERS, cv=2, n_jobs=-1, verbose=1, random_state=42
)
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    xgb_dist, n_iter=XGB_ITERS, cv=2, n_jobs=-1, verbose=1, random_state=42
)

rf_search.fit(X_tr, y_tr)
print("RF best params:", rf_search.best_params_)
dt_search.fit(X_tr, y_tr)
print("DT best params:", dt_search.best_params_)
xgb_search.fit(X_tr, y_tr)
print("XGB best params:", xgb_search.best_params_)

# 10) Save tuned models
joblib.dump(rf_search.best_estimator_, "./models/random_forest_model.pkl")
joblib.dump(dt_search.best_estimator_, "./models/decision_tree_model.pkl")
joblib.dump(xgb_search.best_estimator_, "./models/xgboost_model.pkl")

# 11) Quick cross-validation on full train set
print("\nCross-val scores on subsampled train:")
for name, model in [
    ("Random Forest", rf_search.best_estimator_),
    ("Decision Tree", dt_search.best_estimator_),
    ("XGBoost", xgb_search.best_estimator_)
]:
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# 12) Stacking ensemble
stack = StackingClassifier(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('dt', dt_search.best_estimator_),
        ('xgb', xgb_search.best_estimator_)
    ],
    final_estimator=LogisticRegression(max_iter=200)
)
stack.fit(X_train_tfidf, y_train)

# 13) Evaluate on subsampled test set
def evaluate(m, X, y, label):
    preds = m.predict(X)
    print(f"\n--- {label} ---")
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))

evaluate(rf_search.best_estimator_, X_test_tfidf, y_test,  "Random Forest")
evaluate(dt_search.best_estimator_, X_test_tfidf, y_test,  "Decision Tree")
evaluate(xgb_search.best_estimator_, X_test_tfidf, y_test, "XGBoost")
evaluate(stack,                 X_test_tfidf, y_test, "Stacked Model")

print("\n Training test completed successfully!")
