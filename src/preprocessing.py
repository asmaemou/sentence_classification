import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import joblib

# Create a directory to save the figures for preprocessing
figures_dir = "./preprocessed_data"
os.makedirs(figures_dir, exist_ok=True)

# Load the cleaned dataset (after EDA)
df = pd.read_csv("../data/Amazon_reviews/train_cleaned.csv")

# Handle missing values in the 'clean_review' column (fill NaNs with empty strings)
df["clean_review"].fillna("", inplace=True)

# Handle missing values in 'review_title' column (fill with an empty string or placeholder)
df["review_title"].fillna("No title", inplace=True)

# Handle outliers based on review_length
Q1 = df["review_length"].quantile(0.25)
Q3 = df["review_length"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers based on review length
df_cleaned = df[(df["review_length"] >= lower_bound) & (df["review_length"] <= upper_bound)]
print(f"Dataset size after removing outliers: {df_cleaned.shape[0]}")

# Visualize the outlier distribution after removal
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_cleaned["review_length"])
plt.title("Box Plot of Review Lengths (After Outlier Removal)")
plt.xlabel("Number of Words")
after_outlier_removal_path = os.path.join(figures_dir, "after_outlier_removal.png")
plt.savefig(after_outlier_removal_path)
plt.close()

# Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # Lowercase the text
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Clean the review_text column
df_cleaned["clean_review"] = df_cleaned["review_text"].apply(clean_text)

# Visualize the distribution of cleaned review lengths
df_cleaned["clean_review_length"] = df_cleaned["clean_review"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 5))
sns.histplot(df_cleaned["clean_review_length"], bins=50, kde=True)
plt.title("Cleaned Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")
cleaned_review_length_dist_path = os.path.join(figures_dir, "cleaned_review_length_distribution.png")
plt.savefig(cleaned_review_length_dist_path)
plt.close()

# Save the cleaned data to a new file
train_cleaned_path = "../data/Amazon_reviews/train_cleaned_final.csv"
test_cleaned_path = "../data/Amazon_reviews/test_cleaned_final.csv"

# Split the cleaned data into training and test sets
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned["sentiment"], random_state=42)

# Save cleaned train and test data
train_df.to_csv(train_cleaned_path, index=False)
test_df.to_csv(test_cleaned_path, index=False)

# Check if files exist
print("Checking if files exist:")
print("Train file exists:", os.path.exists(train_cleaned_path))
print("Test file exists:", os.path.exists(test_cleaned_path))

print("\n Data preprocessing completed successfully!")
