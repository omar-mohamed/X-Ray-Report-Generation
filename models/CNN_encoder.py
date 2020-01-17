import tensorflow as tf
from utility import get_layers

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, tags_reducer_units,encoder_layers, tags_embeddings=None):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        if tags_embeddings is not None:
            self.tags_embeddings = tf.keras.layers.Embedding(tags_embeddings.shape[0], tags_embeddings.shape[1],
                                                             weights=[tags_embeddings], trainable=True)
        else:
            self.tags_embeddings = tf.keras.layers.Embedding(105, 400)
        if tags_reducer_units > 0:
            self.tags_reducer = tf.keras.layers.Dense(tags_reducer_units)
        self.tags_reducer_units = tags_reducer_units
        self.encoder_layers=get_layers(encoder_layers, 'relu')
        # self.fc = tf.keras.layers.Dense(embedding_dim, activation = 'relu')

    def call(self, visual_features, tags_predictions = None):

        if tags_predictions is not None:
            tags_predictions = self.tags_embeddings(tags_predictions)
            tags_predictions = tf.reshape(tags_predictions, [tags_predictions.shape[0], tags_predictions.shape[1],
                                                             tags_predictions.shape[2] * tags_predictions.shape[3]])
            if self.tags_reducer_units > 0:
                tags_predictions = self.tags_reducer(tags_predictions)
            features = tf.concat([visual_features, tags_predictions], axis=2)
        else:
            features = visual_features

        for layer in self.encoder_layers:
            features = layer(features)

        # features = self.fc(features)

        return features
