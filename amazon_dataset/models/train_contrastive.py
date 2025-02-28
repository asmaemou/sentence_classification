import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_contrastive_pairs(x_text, y, num_pairs=1000):
    """
    Creates pairs of (textA, textB) with label=1 if same sentiment, 0 otherwise.
    num_pairs: how many pairs to create (limit to keep training time reasonable).
    Split your dataset by sentiment (positive vs. negative).
    Randomly pick pairs from the same sentiment for positive pairs (label=1).
    Randomly pick pairs from different sentiments for negative pairs (label=0).
    """
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

"""
Tokenize and Pad Pairs
Once you have (textA, textB, label) lists, you must tokenize and pad both textA and textB. 
You can re-use your existing tokenizer logic:



"""

def prepare_siamese_data(textA, textB, tokenizer, max_len=100):
    # Convert each list of sentences to sequences
    seqA = tokenizer.texts_to_sequences(textA)
    seqB = tokenizer.texts_to_sequences(textB)

    # Pad them
    padA = pad_sequences(seqA, maxlen=max_len, padding='post')
    padB = pad_sequences(seqB, maxlen=max_len, padding='post')
    return padA, padB
