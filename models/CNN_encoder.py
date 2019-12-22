import tensorflow as tf
import numpy as np


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, tags_embeddings=None):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        if tags_embeddings is not None:
            self.tags_embeddings = tf.keras.layers.Embedding(tags_embeddings.shape[0], tags_embeddings.shape[1],
                                                             weights=[tags_embeddings], trainable=True)
        else:
            self.tags_embeddings = tf.keras.layers.Embedding(105, 400)

        self.tags_reducer = tf.keras.layers.Dense(1024)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.fc2 = tf.keras.layers.Dense(embedding_dim)

    def call(self, tags_predictions, visual_features):
        tags_predictions = self.tags_embeddings(tags_predictions)
        tags_predictions = tf.reshape(tags_predictions, [tags_predictions.shape[0], tags_predictions.shape[1],
                                                         tags_predictions.shape[2] * tags_predictions.shape[3]])
        tags_predictions = self.tags_reducer(tags_predictions)
        concat_features = tf.concat([visual_features, tags_predictions], axis=2)
        x = self.fc(concat_features)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        return x
