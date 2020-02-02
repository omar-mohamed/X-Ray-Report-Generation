import tensorflow as tf
from utility import get_layers


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, encoder_layers, tags_embeddings=None):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        if tags_embeddings is not None:
            self.tags_embeddings = tf.keras.layers.Dense(input_dim=tags_embeddings.shape[0],
                                                         units=tags_embeddings.shape[1], use_bias=False,
                                                         weights=[tags_embeddings], trainable=True)
        else:
            self.tags_embeddings = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.encoder_layers = get_layers(encoder_layers, 'relu')

    def call(self, visual_features, tags_predictions=None):

        if tags_predictions is not None:
            embeddings = self.tags_embeddings(tags_predictions)
            embeddings_sum = tf.reduce_sum(tags_predictions, axis=-1)
            embeddings_sum = tf.reshape(tf.maximum(embeddings_sum, 1),
                                        (embeddings_sum.shape[0], 1, embeddings_sum.shape[-1]))
            embeddings = tf.math.divide(embeddings, embeddings_sum)
            features = tf.concat([visual_features, embeddings], axis=2)
        else:
            features = visual_features

        for layer in self.encoder_layers:
            features = layer(features)

        return features
