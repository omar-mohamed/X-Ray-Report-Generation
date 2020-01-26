import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from models.RNN_decoder import RNN_Decoder
from chexnet_wrapper import ChexnetWrapper
import os
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from caption_evaluation import get_evalutation_scores
from utility import get_enqueuer
import numpy as np
from PIL import Image
import json
import time
from beam_search.beam_path import BeamPath
from beam_search.beam_paths import BeamPaths
from copy import deepcopy


def find_k_largest(x, k, allow_end_seq=True, end_seq_token=0):
    x = np.array(x)
    if not allow_end_seq:
        x[end_seq_token] = 0
    ind = np.argpartition(x, -k)[-k:]
    return ind[np.argsort(tf.gather(x, ind))]


def evaluate_beam_search(FLAGS, encoder, decoder, tokenizer_wrapper, tag_predictions, visual_features, k):
    hidden = decoder.reset_state(batch_size=1)
    features = encoder(visual_features, tag_predictions)
    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")], 0)
    predictions, hidden, _ = decoder(dec_input, features, hidden)
    predictions = tf.nn.softmax(tf.cast(predictions[0], dtype=tf.float64))
    k_largest_ind = find_k_largest(predictions, k, False, tokenizer_wrapper.get_token_of_word("endseq"))
    beam_paths = BeamPaths(k)
    for i in range(k):
        beam_paths.add_path(BeamPath(tokenizer_wrapper, FLAGS.max_sequence_length, [k_largest_ind[i]], hidden,
                                     [predictions[k_largest_ind[i]]]))
    while not beam_paths.should_stop():
        beam_paths.sort()
        hidden = beam_paths.get_best_paths_hidden()
        dec_input = beam_paths.get_best_paths_input()
        best_paths = beam_paths.get_best_k()

        beam_paths.pop_best_k()

        print("________________________")
        for path in best_paths:
            print("Path: {}".format(path.get_sentence_words()))
        # new_paths = []
        t = time.time()
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        print("time taken to predict: {}".format(time.time()-t))
        t = time.time()

        for i in range(predictions.shape[0]):
            preds = tf.nn.softmax(tf.cast(predictions[i], dtype=tf.float64))
            # k_largest_ind = find_k_largest(preds, k)
            for index in range(preds.shape[0]):

                new_path = deepcopy(best_paths[i])
                new_path.add_record(index, preds[index], tf.expand_dims(hidden[i], 0))
                beam_paths.add_path(new_path)
                # new_paths.append(new_path)
        # beam_paths.add_top_k_paths(new_paths)
        print("time taken to add paths: {}".format(time.time()-t))

    # best_paths = beam_paths.get_ended_paths()
    # for path in best_paths:
    #     print(path.get_sentence_words())
    #     print(path.get_prob_list())
    #     print(path.get_total_probability())
    #     print("--------")
    # print("____________________________________")

    return best_paths[0].get_sentence_words()


def evaluate(FLAGS, encoder, decoder, tokenizer_wrapper, tag_predictions, visual_features):
    attention_plot = np.zeros((FLAGS.max_sequence_length, 512))

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(visual_features, tag_predictions)
    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")], 0)
    result = []

    for i in range(FLAGS.max_sequence_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        # predicted_id = tf.argmax(predictions[0]).numpy()
        softmax_predictions = tf.nn.softmax(tf.cast(predictions[0], dtype=tf.float64))
        predicted_id = 1
        counter = 0
        while (predicted_id == 1 and counter < 10) or (
                tokenizer_wrapper.get_word_from_token(predicted_id) == 'endseq' and len(result) == 0):
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


def save_output_prediction(FLAGS, img_name, target_sentence, predicted_sentence):
    if not os.path.exists(FLAGS.output_images_folder):
        os.makedirs(FLAGS.output_images_folder)

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


def evaluate_enqueuer(enqueuer, steps, FLAGS, encoder, decoder, tokenizer_wrapper, chexnet, name='Test set',
                      verbose=True, write_json=True, write_images=False, test_mode=False, beam_search_k=1):
    hypothesis = []
    references = []
    if not enqueuer.is_running():
        enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
    start = time.time()

    generator = enqueuer.get()
    for batch in range(steps):
        if verbose and batch > 0 and batch % 1 == 0:
            print("Step: {}".format(batch))
        t = time.time()
        img, target, img_path = next(generator)
        # print("Time to get batch: {} s ".format(time.time() - t))
        #

        tag_predictions, visual_feaures = chexnet.get_visual_features(img, FLAGS.tags_threshold)
        # print("Time to get visual features: {} s ".format(time.time() - t))

        if not FLAGS.tags_attention:
            tag_predictions = None
        # t = time.time()
        if beam_search_k > 1:
            result = evaluate_beam_search(FLAGS, encoder, decoder, tokenizer_wrapper, tag_predictions, visual_feaures,
                                          beam_search_k)
        else:
            result, attention_plot = evaluate(FLAGS, encoder, decoder, tokenizer_wrapper, tag_predictions,
                                              visual_feaures)
        # print("Time to evaluate step: {} s ".format(time.time() - t))

        target_word_list = tokenizer_wrapper.get_sentence_from_tokens(target)
        references.append([target_word_list])
        hypothesis.append(result)
        target_sentence = tokenizer_wrapper.get_string_from_word_list(target_word_list)
        predicted_sentence = tokenizer_wrapper.get_string_from_word_list(result)
        # t = time.time()
        if write_images:
            save_output_prediction(FLAGS, img_path[0], target_sentence, predicted_sentence)
        # print('Time taken for saving image {} sec\n'.format(time.time() - t))

    enqueuer.stop()
    scores = get_evalutation_scores(hypothesis, references, test_mode)
    print("{} scores: {}".format(name, scores))
    if write_json:
        with open(os.path.join(FLAGS.ckpt_path, 'scores.json'), 'w') as fp:
            json.dump(str(scores), fp, indent=4)
    print('Time taken for evaluation {} sec\n'.format(time.time() - start))

    return scores


if __name__ == "__main__":
    FLAGS = argHandler()
    FLAGS.setDefaults()

    tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                         FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

    print("** load test generator **")

    test_enqueuer, test_steps = get_enqueuer(FLAGS.test_csv, 1, FLAGS, tokenizer_wrapper)
    test_enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)

    encoder = CNN_Encoder(FLAGS.embedding_dim, FLAGS.tags_reducer_units, FLAGS.encoder_layers)
    decoder = RNN_Decoder(FLAGS.embedding_dim, FLAGS.units, FLAGS.tokenizer_vocab_size, FLAGS.classifier_layers)

    optimizer = tf.keras.optimizers.Adam()

    chexnet = ChexnetWrapper('pretrained_models', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers)

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
    evaluate_enqueuer(test_enqueuer, test_steps, FLAGS, encoder, decoder, tokenizer_wrapper, chexnet,
                      write_images=True, test_mode=True, beam_search_k=3)

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
