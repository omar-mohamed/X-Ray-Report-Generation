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
        weights_dir = cp["DEFAULT"].get("weights_dir")
        base_model_name = cp["DEFAULT"].get("base_model_name")
        chexnet_class_names = cp["DEFAULT"].get("chexnet_class_names").split(",")

        # parse weights file path
        output_weights_name = cp["TRAIN"].get("output_weights_name")
        weights_path = os.path.join(weights_dir, output_weights_name)

        print("** load model **")

        model_weights_path = weights_path
        model_factory = ModelFactory()
        self.model = model_factory.get_model(
            chexnet_class_names,
            model_name=base_model_name,
            use_base_weights=False,
            weights_path=model_weights_path,
            pop_last_layer=True)
        self.model.summary()

    def get_visual_features(self, image):

        visual_features = self.model.predict(image)
        return visual_features
