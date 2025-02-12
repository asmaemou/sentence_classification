import pandas as pd

dataset_path = "../data/Amazon_reviews/train.csv"
df = pd.read_csv(dataset_path)

# Print 10 random rows to check sentiment interpretation
print(df.sample(10))
