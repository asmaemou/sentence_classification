import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.text_cnn import TextCNN
# from MR_train.text_rnn import TextRNN
from utils import data_helpers

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "rnn"], help="Choose model type: cnn or rnn")
parser.add_argument("--embedding_type", type=str, default="word2vec", choices=["word2vec", "glove", "fasttext"], help="Choose embedding type")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
args = parser.parse_args()

def preprocess():
    print("Loading dataset...")
    x_text, y = data_helpers.load_data_and_labels(
        "../data/rt-polaritydata/rt-polarity.pos", 
        "../data/rt-polaritydata/rt-polarity.neg"
    )

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_text)
    x = tokenizer.texts_to_sequences(x_text)
    x = pad_sequences(x, maxlen=50)  # Adjust max length if needed

    embedding_matrix = data_helpers.load_embedding_matrix(tokenizer, embedding_type=args.embedding_type, embedding_dim=300)

    return x, y, tokenizer, embedding_matrix

def train():
    print("Training started...")
    x, y, tokenizer, embedding_matrix = preprocess()

    if args.model_type == "cnn":
        model = TextCNN(sequence_length=x.shape[1], num_classes=y.shape[1],
                        vocab_size=len(tokenizer.word_index) + 1, embedding_size=300,
                        extra_dim=50, embedding_matrix=embedding_matrix,
                        filter_sizes=[3, 4, 5], num_filters=100)
    elif args.model_type == "rnn":
        model = TextRNN(sequence_length=x.shape[1], num_classes=y.shape[1],
                        vocab_size=len(tokenizer.word_index) + 1, embedding_size=300,
                        embedding_matrix=embedding_matrix)


    # ✅ Learning Rate Scheduling (More Conservative Decay)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,  # 🔹 Lower Initial Learning Rate
        decay_steps=500,               # 🔹 Faster Decay
        decay_rate=0.90,               # 🔹 Reduce LR More Aggressively
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Stop training when validation loss stops improving
    patience=5,  # Number of epochs to wait before stopping
    restore_best_weights=True  # Restore the best model weights after stopping
)
    model.fit(x, y, batch_size=args.batch_size, epochs=args.num_epochs, validation_split=0.1,callbacks=[early_stopping])

if __name__ == "__main__":
    train()
