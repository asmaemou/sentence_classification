#!/usr/bin/env python3
"""
eda.py

This script cleans the Amazon review dataset before it is used for training.
It:
    - Loads the raw dataset (assumes a CSV file with a "review" column)
    - Removes HTML tags, converts text to lowercase, and removes non-alphabetic characters
    - Drops rows with empty or very short reviews and removes duplicates
    - Saves the cleaned data to a new CSV file
"""

import re
import argparse
import pandas as pd
from bs4 import BeautifulSoup

def clean_text(text):
    """
    Cleans a single text string:
        - Removes HTML tags
        - Converts to lowercase
        - Removes punctuation and numbers (keeps only alphabets and whitespace)
        - Strips extra whitespace
    """
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabet characters (you can adjust this regex as needed)
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_dataset(input_csv, output_csv, text_column="review", min_length=10):
    """
    Cleans the dataset by:
        - Loading the CSV file
        - Dropping rows with missing or empty text
        - Applying text cleaning to the specified text column
        - Removing reviews that are too short after cleaning
        - Dropping duplicate reviews
        - Saving the cleaned dataset to output_csv
    """
    # Load dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    if text_column not in df.columns:
        raise ValueError(f"Expected a '{text_column}' column in the dataset. Found columns: {list(df.columns)}")
    
    # Ensure text is string and strip any surrounding whitespace
    df[text_column] = df[text_column].astype(str).apply(lambda x: x.strip())
    # Drop rows where the text is empty
    df = df[df[text_column] != ""]
    print(f"Rows after dropping empty '{text_column}': {len(df)}")
    
    # Clean the review text and store in a new column 'cleaned_review'
    df['cleaned_review'] = df[text_column].apply(clean_text)
    
    # Remove rows where cleaned text is too short (could be noise)
    df = df[df['cleaned_review'].str.len() >= min_length]
    print(f"Rows after filtering short reviews (<{min_length} characters): {len(df)}")
    
    # Drop duplicate cleaned reviews
    df = df.drop_duplicates(subset=['cleaned_review'])
    print(f"Rows after dropping duplicate reviews: {len(df)}")
    
    # Optionally, you can drop the original review column if you only want the cleaned version:
    # df = df.drop(columns=[text_column])
    
    # Save the cleaned data
    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the Amazon Review Dataset")
    parser.add_argument("--input_csv", type=str, default="../data/Amazon_reviews/train.csv",
                        help="Path to the input CSV file with raw reviews.")
    parser.add_argument("--output_csv", type=str, default="../data/Amazon_reviews/train_cleaned.csv",
                        help="Path to save the cleaned CSV file.")
    parser.add_argument("--text_column", type=str, default="review",
                        help="Name of the column containing the review text.")
    parser.add_argument("--min_length", type=int, default=10,
                        help="Minimum length (in characters) for a cleaned review to be kept.")
    
    args = parser.parse_args()
    clean_dataset(args.input_csv, args.output_csv, args.text_column, args.min_length)
