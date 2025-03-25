#data_helpers.py
import numpy as np
import pandas as pd
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

def clean_str(string: str, lower=True) -> str:
    """
    Cleans input text (optionally lowercasing).
    """
    return string.strip().lower() if lower else string.strip()

def load_data_and_labels(dataset_path, lower=True):
    df = pd.read_csv(dataset_path)
    df["clean_review"] = df["clean_review"].fillna("")
    x_text = df["clean_review"].astype(str).values
    y = df["sentiment"].astype(np.int32).values
    
    return x_text, y


def load_embedding_matrix(tokenizer, embedding_type="word2vec", embedding_dim=300, max_vocab=50000):
    """
    Loads pre-trained Word2Vec, GloVe, or FastText embeddings.

    Args:
        tokenizer: Keras tokenizer object.
        embedding_type: "word2vec", "glove", or "fasttext".
        embedding_dim: Embedding size.
        max_vocab: Maximum vocabulary size.

    Returns:
        Numpy array representing the embedding matrix.
    """
    embedding_index = {}
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)

    if embedding_type == "word2vec":
        embedding_file = "../data/GoogleNews-vectors-negative300.bin"
        print(f"Loading Word2Vec embeddings from {embedding_file}...")
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)

        for word in word2vec_model.index_to_key:
            embedding_index[word] = word2vec_model[word]

    elif embedding_type == "glove":
        embedding_file = "../data/glove.6B.300d.txt"
        print(f"Loading GloVe embeddings from {embedding_file}...")

        with open(embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embedding_index[word] = vector

    elif embedding_type == "fasttext":
        embedding_file = "../data/fasttext.vec"
        print(f"Loading FastText embeddings from {embedding_file}...")

        with open(embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embedding_index[word] = vector
    else:
        raise ValueError("Unsupported embedding type! Use 'word2vec', 'glove', or 'fasttext'.")

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= max_vocab:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

    print(f"Loaded {len(embedding_index)} word vectors from {embedding_type}.")
    return embedding_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_type", type=str, default="word2vec",
                        choices=["word2vec", "glove", "fasttext"],
                        help="Which embedding type to load for testing.")
    args = parser.parse_args()

    dataset_path = "../../data/Amazon_reviews/train_cleaned.csv"
    x_text, y = load_data_and_labels(dataset_path, lower=True)
    print("Loaded", len(x_text), "reviews.")
    print("Sample reviews:")
    for i, review in enumerate(x_text[:5]):
        print(f"Review {i+1}: {review}")
    print("Corresponding sentiment labels:", y[:5])

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(x_text)
    print("Tokenizer vocabulary size:", len(tokenizer.word_index))

    # Use the embedding type from command line
    embedding_matrix = load_embedding_matrix(tokenizer,
                                             embedding_type=args.embedding_type,
                                             embedding_dim=300)
    print("Embedding matrix shape:", embedding_matrix.shape)

