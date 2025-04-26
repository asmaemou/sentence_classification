import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 0) Create output directory for figures
figures_dir = "./preprocessed_data"
os.makedirs(figures_dir, exist_ok=True)

# 1) Load the raw datasets
train_raw = pd.read_csv("../data/Amazon_reviews/train.csv")
test_raw  = pd.read_csv("../data/Amazon_reviews/test.csv")

# 2) Fill missing values & compute review_length on both
for df in (train_raw, test_raw):
    df["review_text"].fillna("", inplace=True)
    df["review_title"].fillna("No title", inplace=True)
    df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))

# 3) Compute IQR bounds from the training set and remove outliers there
Q1 = train_raw["review_length"].quantile(0.25)
Q3 = train_raw["review_length"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_cleaned = train_raw[
    (train_raw["review_length"] >= lower_bound) &
    (train_raw["review_length"] <= upper_bound)
]
# Apply same bounds to test set:
test_cleaned = test_raw[
    (test_raw["review_length"] >= lower_bound) &
    (test_raw["review_length"] <= upper_bound)
]

print(f"Train: before {train_raw.shape[0]}, after {train_cleaned.shape[0]} rows")
print(f"Test:  before {test_raw.shape[0]}, after {test_cleaned.shape[0]} rows")

# 4) Visualize box plot for train after outlier removal
plt.figure(figsize=(10,5))
sns.boxplot(x=train_cleaned["review_length"])
plt.title("Review Lengths After Outlier Removal (Train)")
plt.xlabel("Number of Words")
plt.savefig(os.path.join(figures_dir, "after_outlier_removal_train.png"))
plt.close()

# 5) Define text‐cleaning function
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return ' '.join([t for t in tokens if t not in stop_words])

# 6) Apply cleaning to create `clean_review` and compute its length
for df in (train_cleaned, test_cleaned):
    df["clean_review"] = df["review_text"].apply(clean_text)
    df["clean_review_length"] = df["clean_review"].apply(lambda x: len(x.split()))

# 7) Visualize cleaned‐review length distribution (Train)
plt.figure(figsize=(8,5))
sns.histplot(train_cleaned["clean_review_length"], bins=50, kde=True)
plt.title("Cleaned Review Length Distribution (Train)")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.savefig(os.path.join(figures_dir, "cleaned_review_length_dist_train.png"))
plt.close()

# 8) Save cleaned datasets
train_cleaned.to_csv("../data/Amazon_reviews/train_cleaned_final.csv", index=False)
test_cleaned.to_csv("../data/Amazon_reviews/test_cleaned_final.csv", index=False)

print("\nData preprocessing completed successfully")