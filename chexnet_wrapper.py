from utility import load_model
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf


class ChexnetWrapper:
    def __init__(self, model_path, model_name, pop_conv_layers):
        model = load_model(model_path, model_name)
        self.model = Model(inputs=model.input, outputs=[model.output, model.layers[-pop_conv_layers - 1].output])
        self.model.summary()

    def get_visual_features(self, images, threshold):
        state = tf.keras.backend.learning_phase()
        tf.keras.backend.set_learning_phase(0)
        predictions, visual_features = self.model.predict(images)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        visual_features = np.reshape(visual_features, [visual_features.shape[0], -1])
        predictions = np.reshape(predictions, (predictions.shape[0], -1, predictions.shape[-1]))
        visual_features = np.reshape(visual_features, (visual_features.shape[0], -1, visual_features.shape[-1]))
        predictions = np.array(predictions >= threshold, dtype=np.float32)
        tf.keras.backend.set_learning_phase(state)
        return predictions, visual_features
