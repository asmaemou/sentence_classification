import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_contrastive_pairs(x_text, y, num_pairs=1000):
    # Separate positive & negative indices
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    pairs_textA = []
    pairs_textB = []
    labels = []

    # half of pairs = same sentiment, half = different
    half_pairs = num_pairs // 2

    # 1) Positive pairs (label=1)
    for _ in range(half_pairs):
        i = random.choice(pos_indices)
        j = random.choice(pos_indices)
        pairs_textA.append(x_text[i])
        pairs_textB.append(x_text[j])
        labels.append(1)

    # 2) Negative pairs (label=0)
    for _ in range(half_pairs):
        i = random.choice(pos_indices)
        j = random.choice(neg_indices)
        pairs_textA.append(x_text[i])
        pairs_textB.append(x_text[j])
        labels.append(0)

    # Shuffle pairs
    combined = list(zip(pairs_textA, pairs_textB, labels))
    random.shuffle(combined)
    pairs_textA, pairs_textB, labels = zip(*combined)

    return list(pairs_textA), list(pairs_textB), list(labels)

def prepare_siamese_data(textA, textB, tokenizer, max_len=100):
    # Convert each list of sentences to sequences
    seqA = tokenizer.texts_to_sequences(textA)
    seqB = tokenizer.texts_to_sequences(textB)

    # Pad them
    padA = pad_sequences(seqA, maxlen=max_len, padding='post')
    padB = pad_sequences(seqB, maxlen=max_len, padding='post')
    return padA, padB
if __name__ == '__main__':
    # Dummy data for testing
    x_text = np.array([
        "This movie is great",
        "I did not like this movie",
        "Amazing film, I loved it",
        "Terrible movie, waste of time",
        "Not bad, could be better"
    ])
    # Assign binary labels (e.g., 1 for positive, 0 for negative)
    y = np.array([1, 0, 1, 0, 0])
    
    # Test create_contrastive_pairs function
    pairsA, pairsB, labels = create_contrastive_pairs(x_text, y, num_pairs=4)
    print("Contrastive pairs:")
    for a, b, l in zip(pairsA, pairsB, labels):
        print(f"Pair: [{a}] - [{b}], Label: {l}")
    
    # Dummy tokenizer for testing purposes
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(x_text)
    
    # Test prepare_siamese_data function
    padA, padB = prepare_siamese_data(pairsA, pairsB, tokenizer, max_len=10)
    print("Padded sequences for pair A:")
    print(padA)
    print("Padded sequences for pair B:")
    print(padB)
