import numpy as np
import gensim

def clean_str(string: str) -> str:
    """
    Tokenization and string cleaning for all datasets.
    """
    return string.strip().lower()

def load_data_and_labels(positive_data_file: str, negative_data_file: str):
    """
    Loads the dataset, cleans the text, and generates labels.
    """
    with open(positive_data_file, "r", encoding="utf-8") as f:
        positive_examples = [s.strip() for s in f.readlines()]
    
    with open(negative_data_file, "r", encoding="utf-8") as f:
        negative_examples = [s.strip() for s in f.readlines()]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.array(positive_labels + negative_labels)

    return x_text, y

def load_embedding_matrix(tokenizer, embedding_type="word2vec", embedding_dim=300, max_vocab=50000):
    """
    Loads pre-trained Word2Vec, GloVe, or FastText embeddings and builds an embedding matrix.
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

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

    print(f"Loaded {len(embedding_index)} word vectors from {embedding_type}.")
    return embedding_matrix
