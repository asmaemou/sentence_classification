import tensorflow as tf
from tensorflow.keras import layers

class TextCNN(tf.keras.Model):
    def __init__(self, sequence_length, num_classes, vocab_size, 
                 embedding_size, extra_dim, filter_sizes, num_filters,
                 embedding_matrix=None, trainable_embedding=False):
        super(TextCNN, self).__init__()

        total_embedding_dim = embedding_size + extra_dim  # Extra dimensions added

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            trainable=trainable_embedding  # Train only unknown words + extra dim
        )

        self.conv_layers = []
        self.batch_norm_layers = []
        self.pool_layers = []

        for filter_size in filter_sizes:
            conv_layer = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                padding='valid',
                activation=None  # Remove activation for better BN results
            )
            self.conv_layers.append(conv_layer)
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
            self.pool_layers.append(tf.keras.layers.GlobalMaxPooling1D())

        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense = tf.keras.layers.Dense(
            num_classes, 
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

    def call(self, x, training=False):
        x = self.embedding(x)

        pooled_outputs = []
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            conv_out = conv(x)
            conv_out = bn(conv_out, training=training)  # Apply Batch Normalization
            conv_out = tf.nn.relu(conv_out)  # Apply activation after BN
            pooled = pool(conv_out)
            pooled_outputs.append(pooled)

        x_concat = tf.concat(pooled_outputs, axis=1)
        x_drop = self.dropout(x_concat, training=training)
        return self.dense(x_drop)
