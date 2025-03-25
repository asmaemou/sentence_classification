# train.py
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns


from models.text_cnn import TextCNN
from utils import data_helpers

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_type", type=str, default="word2vec", choices=["word2vec", "glove", "fasttext", "multi"],
                    help="Choose embedding type or multi-channel mode")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--debug", action="store_true", help="Run on a small subset of the data for debugging")
# New arguments for pooling configuration
parser.add_argument("--pooling_type", type=str, default="global_max", choices=["global_max", "k_max"],
                    help="Choose pooling type")
parser.add_argument("--k_max", type=int, default=1, help="k value for KMaxPooling if pooling_type is 'k_max'")
args = parser.parse_args()

def preprocess():
    print("Loading dataset...")
    dataset_path = "../data/Amazon_reviews/train_cleaned.csv"

    lower_text = args.embedding_type != "word2vec"
    x_text, y = data_helpers.load_data_and_labels(dataset_path, lower=lower_text)

    # --- Option: Use a subset of data in debug mode ---
    if args.debug:
        print("Debug mode active: Using only a subset of the data.")
        x_text = x_text[:20000]  # Adjust the number as needed
        y = y[:20000]
    
    y = y.reshape(-1, 1)

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(x_text)

    # Analyze review length distribution
    review_lengths = [len(text.split()) for text in x_text]
    print("Review Length Statistics:")
    print(f"- Max length: {max(review_lengths)}")
    print(f"- Min length: {min(review_lengths)}")
    print(f"- Average length: {np.mean(review_lengths):.2f}")

    # --- Option: Reduce maximum sequence length ---
    max_seq_length = min(int(np.percentile(review_lengths, 90)), 500)  # changed from 95 to 90
    print(f"Setting sequence length to: {max_seq_length}")

    x = tokenizer.texts_to_sequences(x_text)
    x = pad_sequences(x, maxlen=max_seq_length)

    embedding_matrices = []
    if args.embedding_type == "multi":
        print("Using Multi-Channel CNN with Word2Vec, GloVe, and FastText")
        for emb_type in ["word2vec", "glove", "fasttext"]:
            emb_matrix = data_helpers.load_embedding_matrix(tokenizer, embedding_type=emb_type, embedding_dim=300)
            embedding_matrices.append(emb_matrix)
    else:
        embedding_matrix = data_helpers.load_embedding_matrix(tokenizer, embedding_type=args.embedding_type, embedding_dim=300)
        embedding_matrices.append(embedding_matrix)

    return x, y, tokenizer, embedding_matrices

def train():
    print("Training started...")
    x, y, tokenizer, embedding_matrices = preprocess()
    vocab_size = embedding_matrices[0].shape[0]

    model = TextCNN(
        sequence_length=x.shape[1],
        num_classes=1,
        vocab_size=vocab_size,
        embedding_size=300,
        embedding_matrices=embedding_matrices,
        filter_sizes=[3, 4, 5],
        num_filters=75,  
        multi_channel=(args.embedding_type == "multi"),
        pooling_type=args.pooling_type,
        k_max=args.k_max
    )

    # Use a single learning rate strategy
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,  # Lower initial learning rate
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # More aggressive reduction
        patience=4,   # Reduced patience for LR reduction
        min_lr=1e-6,
        verbose=1,
        min_delta=0.001
    )

    # Add model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        x, y,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_split=0.1,
        callbacks=[reduce_lr, checkpoint],
        shuffle=True
    )

    # Print available metrics first
    print("\nAvailable metrics in history:", history.history.keys())

    # Plot Loss and Accuracy with embedding type in title
    plt.figure(figsize=(15, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({args.embedding_type})')
    plt.legend()
    plt.grid(True)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy ({args.embedding_type})')
    plt.legend()
    plt.grid(True)

    # Save the plot with a unique name based on the embedding type
    plot_filename = f'training_results_{args.embedding_type}.png'
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"✅ Plot saved as: {plot_filename}")

    # Optional: Display the plot
    plt.show()

    # Print final metrics
    final_epoch = len(history.history['loss'])
    print(f"\nTraining completed after {final_epoch} epochs")
    print("\nFinal metrics:")
    for metric in history.history.keys():
        print(f"{metric}: {history.history[metric][-1]:.4f}")

        # Evaluate model and calculate predictions
    y_pred = model.predict(x).ravel()
    y_pred_class = (y_pred > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nF1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({args.embedding_type})')
    cm_plot_filename = f'confusion_matrix_{args.embedding_type}.png'
    plt.savefig(cm_plot_filename)
    print(f"✅ Confusion matrix plot saved as: {cm_plot_filename}")
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred_class))

    # Final summary
    final_epoch = len(history.history['loss'])
    print(f"\nTraining completed after {final_epoch} epochs")
    for metric in history.history.keys():
        print(f"{metric}: {history.history[metric][-1]:.4f}")

if __name__ == "__main__":
    train()