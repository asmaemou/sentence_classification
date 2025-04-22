import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
dataset_path = "../data/Amazon_reviews/train.csv"
df = pd.read_csv(dataset_path)

# here i am creating a directory to save the figures inside 'src'
figures_dir = "./figures"
os.makedirs(figures_dir, exist_ok=True)

# Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()
labels = ["Negative", "Positive"]
colors = ["#ff9999", "#66b3ff"]

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Distribution of Sentiment in Reviews")
plt.axis('equal') 
sentiment_plot_path = os.path.join(figures_dir, "sentiment_distribution.png")
plt.savefig(sentiment_plot_path)
plt.close()

# Basic dataset informations
print("\n Dataset Loaded. Shape:", df.shape)
print("\nFirst few rows:\n", df.head())

# Checking Missing Values
print("\n Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)

# Calculate missing values per column
missing_values = df.isnull().sum()

plt.figure(figsize=(6, 4))

# Creation of a bar plot
sns.barplot(x=missing_values.index, y=missing_values.values, color='skyblue')

# Add labels and title
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.title('Missing Values in Each Column')
plt.xticks(rotation=45)  # Rotate column names if needed

# Save the missing values plot
missing_values_plot_path = os.path.join(figures_dir, "missing_values.png")
plt.savefig(missing_values_plot_path)
plt.close()

# Handling Duplicates
print("\n Checking for duplicate rows...")
duplicates = df.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")

# Review Length Analysis
df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 5))
sns.histplot(df["review_length"], bins=50, kde=True)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")
review_length_dist_path = os.path.join(figures_dir, "review_length_distribution.png")
plt.savefig(review_length_dist_path)
plt.close()

# Outlier Detection (Review Length)
Q1 = df["review_length"].quantile(0.25)
Q3 = df["review_length"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["review_length"] < lower_bound) | (df["review_length"] > upper_bound)]
print(f"\nNumber of Outliers: {outliers.shape[0]}")

# Box plot before removing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["review_length"])
plt.title("Box Plot of Review Lengths")
plt.xlabel("Number of Words")
outlier_boxplot_path = os.path.join(figures_dir, "outlier_boxplot.png")
plt.savefig(outlier_boxplot_path)
plt.close()

print("\n EDA Completed Successfully!")