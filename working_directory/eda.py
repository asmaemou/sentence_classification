# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocess import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load Dataset
dataset_path = "../data/Amazon_reviews/train.csv"
df = pd.read_csv(dataset_path)

print("columns names")
print(df.columns)

# Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()
labels = ["Negative", "Positive"]
colors = ["#ff9999", "#66b3ff"]

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Distribution of Sentiment in Reviews")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Basic dataset info
print("\n🔹 Dataset Loaded. Shape:", df.shape)
print("\nFirst few rows:\n", df.head())

# Handling Missing Values
print("\n🔹 Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)




# Calculate missing values per column
missing_values = df.isnull().sum()

# Set up a figure
plt.figure(figsize=(6, 4))

# Create a bar plot
sns.barplot(x=missing_values.index, y=missing_values.values, color='skyblue')

# Add labels and title
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.title('Missing Values in Each Column')
plt.xticks(rotation=45)  # Rotate column names if needed

# Display the plot
plt.tight_layout()
plt.show()

# Handling Duplicates
print("\n🔹 Checking for duplicate rows...")
duplicates = df.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")

# Rename columns explicitly
df.columns = ["sentiment", "review_title", "review_text"]

# Map sentiment values
df["sentiment"] = df["sentiment"].map({1: 0, 2: 1})
print("\nUpdated Label Values:\n", df["sentiment"].value_counts())

# Review Length Analysis
df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 5))
sns.histplot(df["review_length"], bins=50, kde=True)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.show()

# Parallelized text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def parallel_apply(df, func, num_cores=cpu_count()):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    result = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return result

def clean_chunk(df_chunk):
    df_chunk["clean_review"] = df_chunk["review_text"].apply(clean_text)
    return df_chunk

print("\n🔹 Cleaning review text in parallel (this may take time)...")
df_cleaned = parallel_apply(df, clean_chunk)
print("✅ Text cleaning completed.")

# Most Frequent Words
word_freq = Counter(" ".join(df_cleaned["clean_review"]).split()).most_common(20)
words, counts = zip(*word_freq)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(words), y=list(counts))
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()

# Outlier Detection
Q1 = df_cleaned["review_length"].quantile(0.25)
Q3 = df_cleaned["review_length"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"\nLower Bound: {lower_bound}, Upper Bound: {upper_bound}")

outliers = df_cleaned[(df_cleaned["review_length"] < lower_bound) | (df_cleaned["review_length"] > upper_bound)]
print(f"Number of Outliers: {outliers.shape[0]}")

plt.figure(figsize=(10, 5))
sns.boxplot(x=df_cleaned["review_length"])
plt.title("Box Plot of Review Lengths")
plt.xlabel("Number of Words")
plt.show()

# Remove outliers
df_cleaned = df_cleaned[(df_cleaned["review_length"] >= lower_bound) & (df_cleaned["review_length"] <= upper_bound)]
print(f"\nDataset size after removing outliers: {df_cleaned.shape[0]}")


# Box Plot After Outlier Removal
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_cleaned["review_length"])
plt.title("Box Plot of Review Lengths (After Outlier Removal)")
plt.xlabel("Number of Words")
plt.show()

# Verify min and max review lengths
print(f"Min review length after removal: {df_cleaned['review_length'].min()}")
print(f"Max review length after removal: {df_cleaned['review_length'].max()}")

# Train-test split
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned["sentiment"], random_state=42)
print(f"Training set size: {train_df.shape[0]}")
print(f"Test set size: {test_df.shape[0]}")

# Ensure output directory exists
os.makedirs("../data/Amazon_reviews/", exist_ok=True)

# Save datasets
train_path = "../data/Amazon_reviews/train_cleaned.csv"
test_path = "../data/Amazon_reviews/test_cleaned.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Train-test split completed. Cleaned datasets saved!")

# Check if files exist
print("Checking if files exist:")
print("Train file exists:", os.path.exists(train_path))
print("Test file exists:", os.path.exists(test_path))

print("\n🚀 EDA Completed Successfully!")