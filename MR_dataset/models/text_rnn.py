import tensorflow as tf
from tensorflow.keras import layers

class TextRNN(tf.keras.Model):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, embedding_matrix=None):
        super(TextRNN, self).__init__()

        # Embedding Layer (Pretrained Word2Vec)
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=sequence_length,
            trainable=True
        )

        # 🔹 Stacked BiLSTM for Better Feature Extraction
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        )

        # 🔹 GlobalMaxPooling1D to Reduce Dimensionality
        self.global_pool = layers.GlobalMaxPooling1D()

        # 🔹 Batch Normalization
        self.batch_norm = layers.BatchNormalization()

        # 🔹 Fully Connected Layer
        self.dropout = layers.Dropout(rate=0.6)  # Increased dropout
        self.dense = layers.Dense(
            num_classes, 
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.global_pool(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense(x)
