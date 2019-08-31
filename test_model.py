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

class_names = cp["Captioning_Model"].get("class_names").split(",")
image_source_dir = cp["Data"].get("image_source_dir")
data_dir = cp["Data"].get("data_dir")
all_data_csv = cp['Data'].get('all_data_csv')
testing_csv = cp['Data'].get('testing_set_csv')

image_dimension = cp["Chexnet_Default"].getint("image_dimension")

batch_size = cp["Captioning_Model_Inference"].getint("batch_size")
testing_counts = get_sample_counts(data_dir, cp['Data'].get('testing_set_csv'))

max_sequence_length = cp['Captioning_Model'].getint('max_sequence_length')

# These two variables represent that vector shape
features_shape = cp["Captioning_Model"].getint("features_shape")
attention_features_shape = cp["Captioning_Model"].getint("attention_features_shape")

BUFFER_SIZE = cp["Captioning_Model"].getint("buffer_size")
embedding_dim = cp["Captioning_Model"].getint("embedding_dim")
units = cp["Captioning_Model"].getint("units")

checkpoint_path = cp["Captioning_Model_Train"].get("ckpt_path")

output_images_folder = cp["Captioning_Model_Inference"].get("output_images_folder")

# compute steps
steps = int(testing_counts / batch_size)

print(f"** test: {steps} **")

print("** load test generator **")
tokenizer_wrapper = TokenizerWrapper(os.path.join(data_dir, all_data_csv), class_names[0], max_sequence_length)

data_generator = AugmentedImageSequence(
    dataset_csv_file=os.path.join(data_dir, testing_csv),
    class_names=class_names,
    tokenizer_wrapper=tokenizer_wrapper,
    source_image_dir=image_source_dir,
    batch_size=batch_size,
    target_size=(image_dimension, image_dimension),
    steps=steps,
    shuffle_on_epoch_end=False,
)

vocab_size = tokenizer_wrapper.get_tokenizer_word_index()

medical_w2v = Medical_W2V_Wrapper()
embeddings = medical_w2v.get_embeddings_matrix_for_words(tokenizer_wrapper.get_word_tokens_list())
print(embeddings.shape)
del medical_w2v

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size, embeddings)
optimizer = tf.keras.optimizers.Adam()

with graph_mode():
    chexnet = ChexnetWrapper()

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))


def evaluate(image_tensor):
    attention_plot = np.zeros((max_sequence_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=batch_size)

    features = encoder(image_tensor)

    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")], 0)
    result = []

    for i in range(max_sequence_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        # predicted_id = tf.argmax(predictions[0]).numpy()
        softmax_predictions = tf.nn.softmax(tf.cast(predictions[0], dtype=tf.float64))

        predicted_id = np.random.choice(len(predictions[0]), p=softmax_predictions)

        if tokenizer_wrapper.get_word_from_token(predicted_id) == 'endseq':
            return result, attention_plot

        result.append(tokenizer_wrapper.get_word_from_token(predicted_id))

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

if not os.path.exists(output_images_folder):
    os.makedirs(output_images_folder)


def save_output_prediction(img_name, target_sentence, predicted_sentence):
    image_path = os.path.join(image_source_dir, img_name)

    img = mpimg.imread(os.path.join(image_path))

    caption = "Real caption: {}\n\nPrediction: {}".format(target_sentence, predicted_sentence)
    # plt.ioff()
    fig = plt.figure(figsize=(19.20, 10.80))
    fig.add_axes((.0, .3, .9, .7))
    fig.text(.1, .1, caption, wrap=True)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig(output_images_folder + "/{}".format(img_name))
    plt.close(fig)


hypothesis = []
references = []
for batch in range(data_generator.steps):
    print("Batch: {}".format(batch))
    img, target, img_path = data_generator.__getitem__(batch)
    with graph_mode():
        img_tensor = chexnet.get_visual_features(img)
    result, attention_plot = evaluate(img_tensor)
    target_word_list = tokenizer_wrapper.get_sentence_from_tokens(target)
    references.append([target_word_list])
    hypothesis.append(result)
    target_sentence = tokenizer_wrapper.get_string_from_word_list(target_word_list)
    predicted_sentence = tokenizer_wrapper.get_string_from_word_list(result)
    # save_output_prediction(img_path[0], target_sentence, predicted_sentence)

print(get_bleu_scores(hypothesis, references))

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
