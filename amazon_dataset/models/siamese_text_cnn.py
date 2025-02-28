# models/siamese_text_cnn.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Re-use your existing layers: KMaxPooling, AttentionLayer if needed
# or keep it simpler for the Siamese version.

class SiameseEncoder(tf.keras.Model):
    """
    CNN-based encoder that transforms a sequence of tokens into an embedding vector.
    We'll re-use the same logic from your TextCNN but remove the final Dense layer.
    """
    def __init__(self, 
                 vocab_size,
                 embedding_size,
                 embedding_matrices,
                 filter_sizes,
                 num_filters,
                 multi_channel=False,
                 trainable_embedding=True,
                 pooling_type="global_max",
                 k_max=1,
                 dropout_rate=0.7):
        super(SiameseEncoder, self).__init__()

        self.multi_channel = multi_channel
        self.pooling_type = pooling_type
        self.k_max = k_max

        # Embedding layers
        if multi_channel:
            self.embeddings = []
            for emb_matrix in embedding_matrices:
                embedding_layer = layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[emb_matrix] if emb_matrix is not None else None,
                    trainable=trainable_embedding
                )
                self.embeddings.append(embedding_layer)
            # Learnable channel weights
            self.channel_weights = self.add_weight(
                name="channel_weights",
                shape=(len(self.embeddings),),
                initializer=tf.keras.initializers.Ones(),
                trainable=True
            )
        else:
            self.embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                weights=[embedding_matrices[0]] if embedding_matrices is not None else None,
                trainable=trainable_embedding
            )
        
        self.embedding_dropout = layers.SpatialDropout1D(0.3)

        self.conv_layers = []
        self.batch_norm_layers = []
        self.pool_layers = []

        for filter_size in filter_sizes:
            conv_layer = layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                padding='valid',
                activation=None
            )
            self.conv_layers.append(conv_layer)
            self.batch_norm_layers.append(layers.BatchNormalization())
            if pooling_type == "k_max" and k_max > 1:
                # Reuse your KMaxPooling if you like
                from models.text_cnn import KMaxPooling
                self.pool_layers.append(KMaxPooling(k=k_max))
            else:
                self.pool_layers.append(layers.GlobalMaxPooling1D())

        # We can fuse branches with attention or just do a simple concat
        self.attention = None  # optional
        # self.attention = AttentionLayer() # if you want attention

        self.dropout = layers.Dropout(rate=dropout_rate)
        # No final Dense here. We just output the CNN embedding.

    def call(self, x, training=False):
        if self.multi_channel:
            embedded_outputs = [emb(x) for emb in self.embeddings]
            weights = tf.nn.softmax(self.channel_weights)
            stacked = tf.stack(embedded_outputs, axis=-1)
            x = tf.reduce_sum(stacked * weights, axis=-1)
        else:
            x = self.embedding(x)

        # Apply dropout
        x = self.embedding_dropout(x, training=training)

        conv_outputs = []
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            c = conv(x)
            c = bn(c, training=training)
            c = tf.nn.relu(c)
            c = pool(c)  # Pooling reduces the sequence dimension
            conv_outputs.append(c)

        # Merge convolution outputs
        conv_out = tf.concat(conv_outputs, axis=-1)  # Now conv_out is shape (batch_size, total_features)

        # No additional GlobalAveragePooling1D is needed here.
        print(f"Encoder output shape before returning: {conv_out.shape}")
        return conv_out





class SiameseTextCNN(tf.keras.Model):
    def __init__(self, encoder):
        super(SiameseTextCNN, self).__init__()
        self.encoder = encoder
        self.similarity_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        input_a, input_b = inputs

        # Encode both inputs
        encoded_a = self.encoder(input_a)
        encoded_b = self.encoder(input_b)

        # Debug: Print output shape of encoder
        print(f"Encoded A shape: {encoded_a.shape}")
        print(f"Encoded B shape: {encoded_b.shape}")

        # Compute absolute difference
        l1 = tf.abs(encoded_a - encoded_b)

        # Debug: Ensure the shape is correct before reshaping
        print(f"L1 shape before reshaping: {l1.shape}")
        l1 = tf.reshape(l1, (-1, l1.shape[-1]))
        print(f"L1 shape after reshaping: {l1.shape}")

        # Predict similarity score
        sim_score = self.similarity_dense(l1)
        return sim_score


def build(self, input_shape):
    # Extract feature dimension from encoder output
    single_input_shape = input_shape[0]

    # Ensure encoder is built
    self.encoder.build(single_input_shape)

    # Debug: Print encoder output shape
    print(f"Expected encoder output shape: {single_input_shape[-1]}")

    # Fix similarity_dense input shape
    self.similarity_dense.build((None, single_input_shape[-1]))  


