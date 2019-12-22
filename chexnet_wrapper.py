from utility import load_model
import numpy as np
from tensorflow.keras.models import Model

class ChexnetWrapper:
    def __init__(self,model_path, model_name):
        model = load_model(model_path, model_name)
        self.model = Model(inputs=model.input,outputs=[model.output, model.layers[-2].output])
        self.model.summary()
    def get_visual_features(self, images, threshold):

        predictions, visual_features = self.model.predict(images)
        predictions = np.reshape(predictions,(predictions.shape[0],-1,predictions.shape[-1]))
        visual_features = np.reshape(visual_features,(visual_features.shape[0],-1,visual_features.shape[-1]))
        predictions = np.array(predictions >= threshold,dtype=int)
        return predictions, visual_features
