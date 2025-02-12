import tensorflow as tf
from tensorflow.keras import layers

class TextCNN(tf.keras.Model):
    def __init__(self, sequence_length, num_classes, vocab_size, 
                 embedding_size, filter_sizes, num_filters,
                 embedding_matrices=None, trainable_embedding=True, multi_channel=False):
        super(TextCNN, self).__init__()

        self.multi_channel = multi_channel

        if multi_channel:
            self.embeddings = []
            for emb_matrix in embedding_matrices:
                embedding_layer = tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[emb_matrix] if emb_matrix is not None else None,
                    trainable=trainable_embedding
                )
                self.embeddings.append(embedding_layer)
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                weights=[embedding_matrices[0]] if embedding_matrices is not None else None,
                trainable=trainable_embedding
            )
        
        # Add Spatial Dropout here (applies to embeddings)
        self.embedding_dropout = tf.keras.layers.SpatialDropout1D(0.2)

        self.conv_layers = []
        self.batch_norm_layers = []
        self.pool_layers = []

        for filter_size in filter_sizes:
            conv_layer = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                padding='valid',
                activation=None
            )
            self.conv_layers.append(conv_layer)
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
            self.pool_layers.append(tf.keras.layers.GlobalMaxPooling1D())

        self.dropout = tf.keras.layers.Dropout(rate=0.6)
        self.dense = tf.keras.layers.Dense(
            1,  # Change num_classes to 1
            activation="sigmoid",  # Change activation to sigmoid
            kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )


    def call(self, x, training=False):
        if self.multi_channel:
            embedded_outputs = [embedding(x) for embedding in self.embeddings]
            x = tf.concat(embedded_outputs, axis=-1)
        else:
            x = self.embedding(x)
        
        # Apply Spatial Dropout here
        x = self.embedding_dropout(x, training=training)

        pooled_outputs = []
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            conv_out = conv(x)
            conv_out = bn(conv_out, training=training)
            conv_out = tf.nn.relu(conv_out)
            pooled = pool(conv_out)
            pooled_outputs.append(pooled)

        x_concat = tf.concat(pooled_outputs, axis=1)
        x_drop = self.dropout(x_concat, training=training)
        return self.dense(x_drop)