# import tensorflow as tf
# from tensorflow.keras import layers

# class TextCNN(tf.keras.Model):
#     def __init__(self, sequence_length, num_classes, vocab_size, 
#                  embedding_size, filter_sizes, num_filters,
#                  embedding_matrices=None, trainable_embedding=True, multi_channel=False):
#         super(TextCNN, self).__init__()

#         self.multi_channel = multi_channel

#         if multi_channel:
#             self.embeddings = []
#             for emb_matrix in embedding_matrices:
#                 embedding_layer = tf.keras.layers.Embedding(
#                     input_dim=vocab_size,
#                     output_dim=embedding_size,
#                     weights=[emb_matrix] if emb_matrix is not None else None,
#                     trainable=trainable_embedding
#                 )
#                 self.embeddings.append(embedding_layer)
#         else:
#             self.embedding = tf.keras.layers.Embedding(
#                 input_dim=vocab_size,
#                 output_dim=embedding_size,
#                 weights=[embedding_matrices[0]] if embedding_matrices is not None else None,
#                 trainable=trainable_embedding
#             )
        
#         # Add Spatial Dropout here (applies to embeddings)
#         self.embedding_dropout = tf.keras.layers.SpatialDropout1D(0.3)

#         self.conv_layers = []
#         self.batch_norm_layers = []
#         self.pool_layers = []

#         for filter_size in filter_sizes:
#             conv_layer = tf.keras.layers.Conv1D(
#                 filters=num_filters,
#                 kernel_size=filter_size,
#                 padding='valid',
#                 activation=None
#             )
#             self.conv_layers.append(conv_layer)
#             self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
#             self.pool_layers.append(tf.keras.layers.GlobalMaxPooling1D())

#         self.dropout = tf.keras.layers.Dropout(rate=0.7)
#         self.dense = tf.keras.layers.Dense(
#             1,  # Change num_classes to 1
#             activation="sigmoid",  # Change activation to sigmoid
#             kernel_regularizer=tf.keras.regularizers.l2(0.03)
#         )


#     def call(self, x, training=False):
#         if self.multi_channel:
#             embedded_outputs = [embedding(x) for embedding in self.embeddings]
#             x = tf.concat(embedded_outputs, axis=-1)
#         else:
#             x = self.embedding(x)
        
#         # Apply Spatial Dropout here
#         x = self.embedding_dropout(x, training=training)

#         pooled_outputs = []
#         for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
#             conv_out = conv(x)
#             conv_out = bn(conv_out, training=training)
#             conv_out = tf.nn.relu(conv_out)
#             pooled = pool(conv_out)
#             pooled_outputs.append(pooled)

#         x_concat = tf.concat(pooled_outputs, axis=1)
#         x_drop = self.dropout(x_concat, training=training)
#         return self.dense(x_drop)




import tensorflow as tf
from tensorflow.keras import layers

# --- Custom K-Max Pooling Layer ---
class KMaxPooling(layers.Layer):
    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # inputs shape: (batch, sequence_length, channels)
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=False)[0]
        # Flatten the last two dimensions
        return tf.reshape(top_k, (tf.shape(inputs)[0], -1))

# --- Attention Layer for Fusing Convolution Outputs ---
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, num_conv_branches, num_filters)
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1),
            initializer="random_normal", trainable=True
        )
        # Bias shape: (num_conv_branches, 1)
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, conv_branches, num_filters)
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (batch, conv_branches, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, conv_branches)
        alpha = tf.nn.softmax(e, axis=1)  # (batch, conv_branches)
        alpha = tf.expand_dims(alpha, axis=-1)  # (batch, conv_branches, 1)
        context = tf.reduce_sum(inputs * alpha, axis=1)  # (batch, num_filters)
        return context

# --- TextCNN Model ---
class TextCNN(tf.keras.Model):
    def __init__(self, sequence_length, num_classes, vocab_size, 
                 embedding_size, filter_sizes, num_filters,
                 embedding_matrices=None, trainable_embedding=True, multi_channel=False,
                 pooling_type="global_max", k_max=1):
        super(TextCNN, self).__init__()

        self.multi_channel = multi_channel
        self.pooling_type = pooling_type
        self.k_max = k_max

        # Embedding layers
        if multi_channel:
            # Build one embedding layer per channel
            self.embeddings = []
            for emb_matrix in embedding_matrices:
                embedding_layer = layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[emb_matrix] if emb_matrix is not None else None,
                    trainable=trainable_embedding
                )
                self.embeddings.append(embedding_layer)
            # Learnable weights for dynamic fusion of channels
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
        
        # Spatial Dropout on embeddings
        self.embedding_dropout = layers.SpatialDropout1D(0.3)

        # Convolution, BatchNorm, and Pooling layers (one branch per filter size)
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
                self.pool_layers.append(KMaxPooling(k=k_max))
            else:
                self.pool_layers.append(layers.GlobalMaxPooling1D())

        # Attention layer to fuse the outputs of different convolution branches
        self.attention = AttentionLayer()

        self.dropout = layers.Dropout(rate=0.7)
        self.dense = layers.Dense(
            1,  # Binary classification output
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(0.03)
        )

    def call(self, x, training=False):
        # --- Embedding & Dynamic Fusion ---
        if self.multi_channel:
            # Compute embeddings for each channel: each output is (batch, seq_len, embedding_size)
            embedded_outputs = [embedding(x) for embedding in self.embeddings]
            # Compute normalized weights for each channel
            weights = tf.nn.softmax(self.channel_weights)
            # Stack embeddings along a new dimension: (batch, seq_len, embedding_size, num_channels)
            stacked_embeddings = tf.stack(embedded_outputs, axis=-1)
            # Weighted sum over the channels
            x = tf.reduce_sum(stacked_embeddings * weights, axis=-1)
        else:
            x = self.embedding(x)
        
        # Apply Spatial Dropout on embeddings
        x = self.embedding_dropout(x, training=training)

        # --- Convolution, BatchNorm, Activation, and Pooling ---
        conv_outputs = []
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            conv_out = conv(x)
            conv_out = bn(conv_out, training=training)
            conv_out = tf.nn.relu(conv_out)
            pooled = pool(conv_out)
            conv_outputs.append(pooled)
        
        # Stack convolution branch outputs: shape (batch, num_branches, num_filters)
        conv_stack = tf.stack(conv_outputs, axis=1)
        # Fuse branches with an attention mechanism
        attention_output = self.attention(conv_stack)
        
        # Dropout and final dense layer
        x_drop = self.dropout(attention_output, training=training)
        return self.dense(x_drop)




