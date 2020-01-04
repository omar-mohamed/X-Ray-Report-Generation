import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from models.RNN_decoder import RNN_Decoder
from chexnet_wrapper import ChexnetWrapper
from configs import argHandler
from generator import AugmentedImageSequence
import time
from augmenter import augmenter
from medical_w2v_wrapper import Medical_W2V_Wrapper
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
from utility import get_optimizer, load_model
import os
import json

FLAGS = argHandler()
FLAGS.setDefaults()

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                     FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

data_generator = AugmentedImageSequence(
    dataset_csv_file= FLAGS.train_csv,
    class_names=FLAGS.csv_label_columns,
    tokenizer_wrapper=tokenizer_wrapper,
    source_image_dir=FLAGS.image_directory,
    batch_size=FLAGS.batch_size,
    target_size=FLAGS.image_target_size,
    augmenter=augmenter,
    shuffle_on_epoch_end=True,
)

medical_w2v = Medical_W2V_Wrapper()
embeddings = medical_w2v.get_embeddings_matrix_for_words(tokenizer_wrapper.get_word_tokens_list(), FLAGS.tokenizer_vocab_size)
tags_embeddings = medical_w2v.get_embeddings_matrix_for_tags(FLAGS.tags)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Tags Embeddings shape: {tags_embeddings.shape}")

del medical_w2v

encoder = CNN_Encoder(FLAGS.embedding_dim, FLAGS.tags_reducer_units, FLAGS.encoder_layers, tags_embeddings)
decoder = RNN_Decoder(FLAGS.embedding_dim, FLAGS.units, FLAGS.tokenizer_vocab_size, FLAGS.classifier_layers, embeddings)


optimizer = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


loss_plot = []


@tf.function
def train_step(tag_predictions, visual_features, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")] * FLAGS.batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(visual_features, tag_predictions)
        # print("encoded")

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    # print("decoded")
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    # print("will apply gradients")
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    # print("applied gradients")
    return loss, total_loss


chexnet = ChexnetWrapper('pretrained_models',FLAGS.visual_model_name, FLAGS.visual_model_pop_layers)

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

try:
    os.makedirs(FLAGS.ckpt_path)
except:
    print("path already exists")

with open(os.path.join(FLAGS.ckpt_path,'configs.json'), 'w') as fp:
    json.dump(FLAGS, fp, indent=4)
ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

start_epoch = 0
if ckpt_manager.latest_checkpoint and FLAGS.continue_from_last_ckpt:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

for epoch in range(start_epoch, FLAGS.num_epochs):
    start = time.time()
    total_loss = 0

    for batch in range(data_generator.steps):
        t = time.time()
        img, target,_ = data_generator.__getitem__(batch)
        print("Time to get batch: {} s ".format(time.time()-t))
        # print( target.max())
        t = time.time()
        tag_predictions, visual_feaures = chexnet.get_visual_features(img, FLAGS.tags_threshold)
        if not FLAGS.tags_attention:
            tag_predictions = None
        print("Time to get visual features: {} s ".format(time.time()-t))

        # img_tensor=np.random.randint(low=-1,high=1,size=(1,1024))
        # img_tensor=np.float32(img_tensor)
        t = time.time()
        batch_loss, t_loss = train_step(tag_predictions, visual_feaures, target)
        total_loss += t_loss
        print("Time to train step: {} s ".format(time.time()-t))

        if batch % 20 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / data_generator.steps)

    if epoch % 1 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss / data_generator.steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig(FLAGS.ckpt_path+"/loss.png")

# plt.show()
