import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from models.RNN_decoder import RNN_Decoder
from chexnet_wrapper import ChexnetWrapper
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from utility import get_sample_counts
from tensorflow.python.eager.context import eager_mode, graph_mode
from medical_w2v_wrapper import Medical_W2V_Wrapper
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from caption_evaluation import get_bleu_scores
import numpy as np
from PIL import Image

config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["DEFAULT"].get("class_names").split(",")
image_source_dir = cp["DEFAULT"].get("image_source_dir")
data_dir = cp["DEFAULT"].get("data_dir")

# train config
image_dimension = cp["TRAIN"].getint("image_dimension")

# test config
BATCH_SIZE = 1
steps = cp["TEST"].get("test_steps")
training_counts = get_sample_counts(data_dir, "testing_set", class_names)
# compute steps
if steps == "auto":
    steps = int(training_counts / BATCH_SIZE)
else:
    try:
        steps = int(steps)
    except ValueError:
        raise ValueError(f"""
            test_steps: {steps} is invalid,
            please use 'auto' or integer.
            """)
print(f"** test: {steps} **")

print("** load test generator **")
max_length = 170
tokenizer_wrapper = TokenizerWrapper(os.path.join(data_dir, "all_data.csv"), class_names[0], max_length)

data_generator = AugmentedImageSequence(
    dataset_csv_file=os.path.join(data_dir, "testing_set.csv"),
    class_names=class_names,
    tokenizer_wrapper=tokenizer_wrapper,
    source_image_dir=image_source_dir,
    batch_size=BATCH_SIZE,
    target_size=(image_dimension, image_dimension),
    steps=steps,
    shuffle_on_epoch_end=True,
)

BUFFER_SIZE = 1000
embedding_dim = 400
units = 512
vocab_size = tokenizer_wrapper.get_tokenizer_word_index()
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 1024
attention_features_shape = 64

medical_w2v = Medical_W2V_Wrapper()
embeddings = medical_w2v.get_embeddings_matrix_for_words(tokenizer_wrapper.get_word_tokens_list())
print(embeddings.shape)
del medical_w2v

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size, embeddings)

with graph_mode():
    chexnet = ChexnetWrapper()

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))


def evaluate(image_tensor):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=BATCH_SIZE)

    features = encoder(image_tensor)

    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer_wrapper.get_token_of_word(predicted_id))

        if tokenizer_wrapper.get_token_of_word(predicted_id) == 'endseq':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


total_loss = 0

output_images_folder = "output_images"
if not os.path.exists(output_images_folder):
    os.makedirs(output_images_folder)


def save_output_prediction(img_path, target_sentence, predicted_sentence):
    img = mpimg.imread(img_path)
    caption = "Real caption: {}\n\nPrediction: {}".format(target_sentence, predicted_sentence)
    fig = plt.figure()
    fig.add_axes((.1, .3, .9, .7))
    fig.text(.3, .1, caption)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.imsave(output_images_folder + "/{}".format(os.path.basename(img_path)))


hypothesis = []
refrences = []
for batch in range(data_generator.steps):
    img, target = data_generator.__getitem__(batch)
    with graph_mode():
        img_tensor = chexnet.get_visual_features(img)
    result, attention_plot = evaluate(img_tensor)
    target_word_list = tokenizer_wrapper.get_sentence_from_tokens(target)
    refrences.append([target_word_list])
    hypothesis.append(result)
    target_sentence = tokenizer_wrapper.get_string_from_word_list(target[1:-1])
    predicted_sentence = tokenizer_wrapper.get_string_from_word_list(result[1:-1])
    save_output_prediction(img, target_sentence, predicted_sentence)

print(get_bleu_scores(hypothesis, refrences))
# # captions on the validation set
# rid = np.random.randint(0, len(img_name_val))
# image = img_name_val[rid]
# real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
# result, attention_plot = evaluate(image)
#
# print ('Real Caption:', real_caption)
# print ('Prediction Caption:', ' '.join(result))
# plot_attention(image, result, attention_plot)
# # opening the image
# Image.open(img_name_val[rid])
