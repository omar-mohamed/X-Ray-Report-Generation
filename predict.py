import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.chexnet import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    weights_dir = cp["DEFAULT"].get("weights_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    chexnet_class_names = cp["DEFAULT"].get("chexnet_class_names").split(",")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    data_dir = cp["DEFAULT"].get("data_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(weights_dir, output_weights_name)

    # get test sample count
    test_counts = get_sample_counts(data_dir, "all_data", class_names)
    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")

    model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        chexnet_class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path,
        pop_last_layer=True)
    model.summary()
    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(data_dir, "all_data.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )

    print("** make prediction **")
    image,y = test_sequence.__getitem__(4)

    y_hat=model.predict(image)
    # y_hat = model.predict_generator(test_sequence, verbose=1)
    # y = test_sequence.get_y_true()

    print(y_hat.shape)

if __name__ == "__main__":
    main()