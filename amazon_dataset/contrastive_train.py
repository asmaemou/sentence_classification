# contrastive_train.py
"""
Train the Siamese Model with Contrastive Loss
We can simply compile with binary crossentropy since we’re predicting similarity_score in [0,1]. 
If label=1 → same sentiment, we want similarity_score ~ 1. 
If label=0 → different sentiment, we want similarity_score ~ 0.

"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from utils.data_helpers import load_data_and_labels, load_embedding_matrix
from models.siamese_text_cnn import SiameseEncoder, SiameseTextCNN
from models.train_contrastive import create_contrastive_pairs, prepare_siamese_data

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_type", type=str, default="word2vec", 
                    choices=["word2vec", "glove", "fasttext", "multi"],
                    help="Which embedding to use?")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--max_pairs", type=int, default=50000, help="How many pairs to create")
parser.add_argument("--debug", action="store_true", help="Run on a small subset of the data for debugging")

args = parser.parse_args()

def main():
    # 1) Load raw data
    dataset_path = "../data/Amazon_reviews/train_cleaned.csv"
    x_text, y = load_data_and_labels(dataset_path, lower=True)

    if args.debug:
        print("Debug mode active: Using only a subset of the data.")
        x_text = x_text[:20000]
        y = y[:20000]

    # 2) Tokenizer
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(x_text)
    vocab_size = min(len(tokenizer.word_index) + 1, 50000)

    # 3) Create pairs for contrastive learning
    textA, textB, labels = create_contrastive_pairs(x_text, y, num_pairs=args.max_pairs)

    # 4) Train/Validation split for pairs
    trainA, valA, trainB, valB, trainY, valY = train_test_split(
        textA, textB, labels, test_size=0.1, random_state=42
    )
    trainA, trainB, trainY = np.array(trainA), np.array(trainB), np.array(trainY).reshape(-1, 1)
    valA, valB, valY = np.array(valA), np.array(valB), np.array(valY).reshape(-1, 1)

    # 5) Prepare data for Siamese model
    max_len = 100
    pad_trainA, pad_trainB = prepare_siamese_data(trainA, trainB, tokenizer, max_len=max_len)
    pad_valA, pad_valB = prepare_siamese_data(valA, valB, tokenizer, max_len=max_len)

    # 6) Load embeddings
    embedding_matrices = []
    if args.embedding_type == "multi":
        for emb_type in ["word2vec", "glove", "fasttext"]:
            emb_matrix = load_embedding_matrix(tokenizer, emb_type, embedding_dim=300)
            embedding_matrices.append(emb_matrix)
    else:
        emb_matrix = load_embedding_matrix(tokenizer, args.embedding_type, embedding_dim=300)
        embedding_matrices.append(emb_matrix)

    # 7) Build Siamese Encoder
    encoder = SiameseEncoder(
        vocab_size=vocab_size,
        embedding_size=300,
        embedding_matrices=embedding_matrices,
        filter_sizes=[3,4,5],
        num_filters=75,
        multi_channel=(args.embedding_type == "multi"),
        pooling_type="global_max",
        k_max=1
    )

    # 8) Build Siamese Model
    siamese_model = SiameseTextCNN(encoder)

    # 9) Explicitly Call and Build Model
    dummy_input = (tf.zeros((1, max_len), dtype=tf.int32), tf.zeros((1, max_len), dtype=tf.int32))
    print(f"Dummy input shapes: {[d.shape for d in dummy_input]}")

    _ = siamese_model(dummy_input, training=False)  # Ensures TensorFlow initializes layers correctly before fitting the model.


    # 10) Compile
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Debugging: Print training data shapes before fitting
    print("pad_trainA shape:", pad_trainA.shape)
    print("pad_trainB shape:", pad_trainB.shape)
    print("trainY shape:", trainY.shape)

    # 11) Train
    history = siamese_model.fit(
        x=[pad_trainA, pad_trainB],
        y=trainY,
        validation_data=([pad_valA, pad_valB], valY),
        batch_size=args.batch_size,
        epochs=args.num_epochs
    )

# Plot training and validation loss & accuracy
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss ({args.embedding_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy {args.embedding_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the training metrics plot with a filename based on the embedding type
    plt.tight_layout()
    metrics_filename = f"{args.embedding_type}_training_metrics.png"
    plt.savefig(metrics_filename)
    plt.close()
    print(f"Saved training metrics plot to {metrics_filename}")

    # Generate predictions on the validation set for confusion matrix calculation
    val_predictions = siamese_model.predict([pad_valA, pad_valB])
    # Convert probabilities to binary predictions using 0.5 as the threshold
    val_predictions = (val_predictions > 0.5).astype(int)

    # Compute the confusion matrix
    cm = confusion_matrix(valY, val_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    # Save the confusion matrix plot
    cm_filename = f"{args.embedding_type}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_filename}")

    # Print final metrics
    print("Training completed.")
    print("Final training accuracy:", history.history['accuracy'][-1])
    if 'val_accuracy' in history.history:
        print("Final validation accuracy:", history.history['val_accuracy'][-1])

if __name__ == "__main__":
    main()
