import os
from configparser import ConfigParser
from models.chexnet import ModelFactory

class ChexnetWrapper:
    def __init__(self):
        # parser config
        config_file = "./config.ini"
        cp = ConfigParser()
        cp.read(config_file)

        # default config
        weights_dir = cp["Chexnet_Default"].get("weights_dir")
        base_model_name = cp["Chexnet_Default"].get("base_model_name")
        chexnet_class_names = cp["Chexnet_Default"].get("chexnet_class_names").split(",")

        # parse weights file path
        weights_name = cp["Chexnet_Inference"].get("weights_name")
        weights_path = os.path.join(weights_dir, weights_name)
        model_weights_path = weights_path
        use_base_weights=cp["Chexnet_Inference"].getboolean("use_base_model_weights")
        print("** load model **")

        model_factory = ModelFactory()
        self.model = model_factory.get_model(
            chexnet_class_names,
            model_name=base_model_name,
            use_base_weights=use_base_weights,
            weights_path=model_weights_path,
            pop_last_layer=True)
        self.model.summary()

    def get_visual_features(self, images):

        visual_features = self.model.predict(images)
        return visual_features
