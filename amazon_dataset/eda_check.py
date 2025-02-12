import pandas as pd

train_df = pd.read_csv("../data/Amazon_reviews/train_cleaned.csv")
test_df = pd.read_csv("../data/Amazon_reviews/test_cleaned.csv")

print("\nTrain Dataset:")
print(train_df.head())

print("\nTest Dataset:")
print(test_df.head())
