import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from models.RNN_decoder import RNN_Decoder
from chexnet_wrapper import ChexnetWrapper
import os
from configs import argHandler
from generator import AugmentedImageSequence
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from caption_evaluation import get_bleu_scores
import numpy as np
from PIL import Image
import json

FLAGS = argHandler()
FLAGS.setDefaults()

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                     FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

print("** load test generator **")

data_generator = AugmentedImageSequence(
    dataset_csv_file=FLAGS.test_csv,
    class_names=FLAGS.csv_label_columns,
    tokenizer_wrapper=tokenizer_wrapper,
    source_image_dir=FLAGS.image_directory,
    batch_size=1,
    target_size=FLAGS.image_target_size,
    shuffle_on_epoch_end=False,
)

encoder = CNN_Encoder(FLAGS.embedding_dim)
decoder = RNN_Decoder(FLAGS.embedding_dim, FLAGS.units, FLAGS.tokenizer_vocab_size)
optimizer = tf.keras.optimizers.Adam()

chexnet = ChexnetWrapper('pretrained_models',FLAGS.visual_model_name, FLAGS.visual_model_pop_layers)

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))


def evaluate(tag_predictions, visual_features):
    attention_plot = np.zeros((FLAGS.max_sequence_length, 512))

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(tag_predictions, visual_features)

    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")], 0)
    result = []

    for i in range(FLAGS.max_sequence_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        # predicted_id = tf.argmax(predictions[0]).numpy()
        softmax_predictions = tf.nn.softmax(tf.cast(predictions[0], dtype=tf.float64))
        predicted_id = 1
        counter = 0
        while predicted_id == 1 and counter < 10:
            predicted_id = np.random.choice(len(predictions[0]), p=softmax_predictions)
            counter += 1
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

if not os.path.exists(FLAGS.output_images_folder):
    os.makedirs(FLAGS.output_images_folder)


def save_output_prediction(img_name, target_sentence, predicted_sentence):
    image_path = os.path.join(FLAGS.image_directory, img_name)

    img = mpimg.imread(os.path.join(image_path))

    caption = "Real caption: {}\n\nPrediction: {}".format(target_sentence, predicted_sentence)
    # plt.ioff()
    fig = plt.figure(figsize=(7.20, 10.80))
    fig.add_axes((.0, .5, .9, .7))
    fig.text(.1, .3, caption, wrap=True, fontsize=20)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig(FLAGS.output_images_folder + "/{}".format(img_name))
    plt.close(fig)


hypothesis = []
references = []
for batch in range(data_generator.steps):
    print("Batch: {}".format(batch))
    img, target, img_path = data_generator.__getitem__(batch)
    tag_predictions, visual_feaures = chexnet.get_visual_features(img, FLAGS.tags_threshold)
    result, attention_plot = evaluate(tag_predictions, visual_feaures)
    target_word_list = tokenizer_wrapper.get_sentence_from_tokens(target)
    references.append([target_word_list])
    hypothesis.append(result)
    target_sentence = tokenizer_wrapper.get_string_from_word_list(target_word_list)
    predicted_sentence = tokenizer_wrapper.get_string_from_word_list(result)
    save_output_prediction(img_path[0], target_sentence, predicted_sentence)

scores = get_bleu_scores(hypothesis, references)
print(scores)
with open(os.path.join(FLAGS.ckpt_path,'scores.json'), 'w') as fp:
    json.dump(scores, fp, indent=4)

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
